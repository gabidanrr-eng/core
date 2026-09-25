"""Operational metrics computed from the durable accounting tables (no separate telemetry store).

Everything here is derived from rows the runtime already writes for every model call, tool call,
task transition, evidence gate, approval and failure, so the numbers are auditable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _r(value: float | None, digits: int = 0) -> float | None:
    return None if value is None else round(value, digits)


def model_metrics(rt: CoreRuntime, since: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for r in rt.db.query(
        "SELECT provider_id, model_key, COALESCE(role, '-') AS role, status, error_class, input_tokens, output_tokens, "
        "cached_tokens, cost_usd, latency_ms, ttft_ms, retries FROM model_calls WHERE started_at >= ?",
        (since,),
    ):
        groups[(r["provider_id"], r["model_key"] or "-", r["role"])].append(r)
    out = []
    for (provider, model, role), rows in sorted(groups.items()):
        errors: dict[str, int] = defaultdict(int)
        for r in rows:
            if r["status"] == "error":
                errors[r["error_class"] or "unknown"] += 1
        latencies = [
            float(r["latency_ms"]) for r in rows if r["latency_ms"] is not None and r["status"] == "ok"
        ]
        ttfts = [float(r["ttft_ms"]) for r in rows if r["ttft_ms"] is not None and r["status"] == "ok"]
        costs = [float(r["cost_usd"]) for r in rows if r["cost_usd"] is not None]
        out.append(
            {
                "provider": provider,
                "model": model,
                "role": role,
                "calls": len(rows),
                "errors": dict(errors),
                "error_rate": round(sum(errors.values()) / len(rows), 3),
                "retries": sum(int(r["retries"] or 0) for r in rows),
                "latency_p50_ms": _r(percentile(latencies, 50)),
                "latency_p95_ms": _r(percentile(latencies, 95)),
                "ttft_p50_ms": _r(percentile(ttfts, 50)),
                "input_tokens": sum(int(r["input_tokens"] or 0) for r in rows),
                "output_tokens": sum(int(r["output_tokens"] or 0) for r in rows),
                "cached_tokens": sum(int(r["cached_tokens"] or 0) for r in rows),
                "cost_usd": round(sum(costs), 6) if costs else None,
            }
        )
    return out


def tool_metrics(rt: CoreRuntime, since: float) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for r in rt.db.query(
        "SELECT tool, status, error_class, duration_ms FROM tool_calls WHERE started_at >= ?", (since,)
    ):
        groups[r["tool"]].append(r)
    out = []
    for tool, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        statuses: dict[str, int] = defaultdict(int)
        for r in rows:
            statuses[r["status"]] += 1
        durations = [float(r["duration_ms"]) for r in rows if r["duration_ms"] is not None]
        out.append(
            {
                "tool": tool,
                "calls": len(rows),
                "statuses": dict(statuses),
                "failure_rate": round(sum(n for s, n in statuses.items() if s != "ok") / len(rows), 3),
                "p50_ms": _r(percentile(durations, 50)),
                "p95_ms": _r(percentile(durations, 95)),
            }
        )
    return out


def task_metrics(rt: CoreRuntime, since: float) -> dict[str, Any]:
    rows = rt.db.query(
        "SELECT kind, mode, status, attempt_count, recovered_count, evidence_level, created_at, completed_at "
        "FROM tasks WHERE created_at >= ? AND parent_task_id IS NULL",
        (since,),
    )
    by_kind: dict[str, dict[str, Any]] = {}
    for r in rows:
        entry = by_kind.setdefault(
            r["kind"],
            {"tasks": 0, "statuses": defaultdict(int), "attempts": 0, "recovered": 0, "durations": []},
        )
        entry["tasks"] += 1
        entry["statuses"][r["status"]] += 1
        entry["attempts"] += int(r["attempt_count"] or 0)
        entry["recovered"] += int(r["recovered_count"] or 0)
        if r["status"] == "completed" and r["completed_at"]:
            entry["durations"].append(float(r["completed_at"]) - float(r["created_at"]))
    kinds = []
    for kind, e in sorted(by_kind.items()):
        kinds.append(
            {
                "kind": kind,
                "tasks": e["tasks"],
                "statuses": dict(e["statuses"]),
                "completion_rate": round(e["statuses"].get("completed", 0) / e["tasks"], 3),
                "avg_attempts": round(e["attempts"] / e["tasks"], 2),
                "recovered": e["recovered"],
                "duration_p50_s": _r(percentile(e["durations"], 50), 1),
            }
        )
    levels: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["status"] == "completed":
            levels[r["evidence_level"] or "none"] += 1
    gates = {
        r["status"]: r["n"]
        for r in rt.db.query(
            "SELECT status, COUNT(*) AS n FROM evidence WHERE kind = 'gate' AND created_at >= ? GROUP BY status",
            (since,),
        )
    }
    return {"total": len(rows), "by_kind": kinds, "evidence_levels": dict(levels), "gates": gates}


def collect(rt: CoreRuntime, *, days: int = 30) -> dict[str, Any]:
    since = rt.clock.now() - days * 86400
    models = model_metrics(rt, since)
    tools = tool_metrics(rt, since)
    tasks = task_metrics(rt, since)
    approvals = {
        r["status"]: r["n"]
        for r in rt.db.query(
            "SELECT status, COUNT(*) AS n FROM approvals WHERE created_at >= ? GROUP BY status", (since,)
        )
    }
    failures = [
        dict(r)
        for r in rt.db.query(
            "SELECT category, error_class, COUNT(*) AS n, SUM(CASE WHEN outcome = 'resolved' THEN 1 ELSE 0 END) AS resolved "
            "FROM failures WHERE created_at >= ? GROUP BY category, error_class ORDER BY n DESC LIMIT 20",
            (since,),
        )
    ]
    data: dict[str, Any] = {
        "days": days,
        "models": models,
        "tools": tools,
        "tasks": tasks,
        "approvals": approvals,
        "failures": failures,
    }
    data["text"] = render(data)
    return data


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.0f}" if value >= 100 else f"{value:g}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    cells = [[_fmt(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in cells]
    return lines


def render(data: dict[str, Any]) -> str:
    lines = [f"Metrics for the last {data['days']} day(s)", ""]
    tasks = data["tasks"]
    lines.append(
        f"Tasks: {tasks['total']} top-level; gates {tasks['gates'] or {}}; evidence levels {tasks['evidence_levels'] or {}}"
    )
    if tasks["by_kind"]:
        lines += _table(
            ["kind", "tasks", "completed", "avg attempts", "recovered", "p50 s"],
            [
                [
                    k["kind"],
                    k["tasks"],
                    f"{k['completion_rate']:.0%}",
                    k["avg_attempts"],
                    k["recovered"],
                    k["duration_p50_s"],
                ]
                for k in tasks["by_kind"]
            ],
        )
    lines.append("")
    lines.append("Model calls:")
    if data["models"]:
        lines += _table(
            [
                "provider",
                "model",
                "role",
                "calls",
                "err%",
                "errors",
                "p50 ms",
                "p95 ms",
                "ttft ms",
                "in tok",
                "out tok",
                "cost $",
            ],
            [
                [
                    m["provider"],
                    m["model"],
                    m["role"],
                    m["calls"],
                    f"{m['error_rate']:.0%}",
                    ",".join(f"{k}:{v}" for k, v in m["errors"].items()) or "-",
                    m["latency_p50_ms"],
                    m["latency_p95_ms"],
                    m["ttft_p50_ms"],
                    m["input_tokens"],
                    m["output_tokens"],
                    None if m["cost_usd"] is None else f"{m['cost_usd']:.4f}",
                ]
                for m in data["models"]
            ],
        )
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Tools:")
    if data["tools"]:
        lines += _table(
            ["tool", "calls", "fail%", "statuses", "p50 ms", "p95 ms"],
            [
                [
                    t["tool"],
                    t["calls"],
                    f"{t['failure_rate']:.0%}",
                    ",".join(f"{k}:{v}" for k, v in t["statuses"].items()),
                    t["p50_ms"],
                    t["p95_ms"],
                ]
                for t in data["tools"][:25]
            ],
        )
    else:
        lines.append("  (none)")
    if data["approvals"]:
        lines += ["", f"Approvals: {data['approvals']}"]
    if data["failures"]:
        lines += ["", "Failures:"]
        lines += _table(
            ["category", "error class", "count", "resolved"],
            [[f["category"], f["error_class"], f["n"], f["resolved"]] for f in data["failures"]],
        )
    return "\n".join(lines)
