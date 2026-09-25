"""Evaluation reports: one run in detail, or a suite's trend with regressions between runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coremain.errors import NotFoundError
from coremain.util.jsonutil import loads

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime


def _run(rt: CoreRuntime, run_id: str) -> dict[str, Any]:
    row = rt.db.one("SELECT * FROM eval_runs WHERE id = ? OR id LIKE ?", (run_id, f"%{run_id}"))
    if row is None:
        raise NotFoundError(f"no evaluation run {run_id}")
    run = dict(row)
    run["summary"] = loads(run.pop("summary_json"), {})
    run["strategy"] = loads(run.pop("strategy_json"), {})
    run["results"] = []
    for r in rt.db.query("SELECT * FROM eval_results WHERE run_id = ? ORDER BY created_at", (run["id"],)):
        item = dict(r)
        item["metrics"] = loads(item.pop("metrics_json"), {})
        item["checks"] = loads(item.pop("checks_json"), [])
        run["results"].append(item)
    return run


def build_report(rt: CoreRuntime, *, run_id: str | None = None, suite: str | None = None) -> dict[str, Any]:
    if run_id is None:
        sql = (
            "SELECT id FROM eval_runs"
            + (" WHERE suite = ?" if suite else "")
            + " ORDER BY started_at DESC LIMIT 10"
        )
        ids = [r["id"] for r in rt.db.query(sql, (suite,) if suite else ())]
        if not ids:
            raise NotFoundError("no evaluation runs recorded yet", hint="run `core eval run --suite smoke`")
        runs = [_run(rt, i) for i in ids]
    else:
        runs = [_run(rt, run_id)]
    latest = runs[0]
    lines = [
        f"Evaluation run {latest['id']} · suite {latest['suite']} · core {latest['core_version']} · prompts {latest['prompt_version']}",
        f"strategy: {latest['strategy']}",
    ]
    s = latest["summary"]
    if s:
        lines.append(
            f"result: {s.get('passed', 0)}/{s.get('total', 0)} passed ({s.get('pass_rate', 0):.0%}), "
            f"{s.get('failed', 0)} failed, {s.get('errors', 0)} errors, {s.get('skipped', 0)} skipped · "
            f"{s.get('model_calls', 0)} model calls, {s.get('tokens', 0)} tokens, {s.get('duration_s', 0)}s"
        )
    lines.append("")
    for r in latest["results"]:
        m = r["metrics"]
        lines.append(
            f"{'PASS' if r['status'] == 'pass' else r['status'].upper():<7} {r['scenario']:<34} {r['family']:<18} "
            f"task={r['task_status'] or '-'} calls={m.get('model_calls', '-')} tools={m.get('tool_calls', '-')} "
            f"attempts={m.get('attempts', '-')} level={m.get('evidence_level') or '-'} {m.get('duration_s', '-')}s"
        )
        for c in r["checks"]:
            if not c["ok"]:
                lines.append(f"        ✗ {c['check']}: {c['detail']}")
        if r.get("error"):
            lines.append(f"        ! {r['error']}")
    regressions: list[str] = []
    if len(runs) > 1:
        previous = next((p for p in runs[1:] if p["suite"] == latest["suite"]), None)
        if previous is not None:
            before = {r["scenario"]: r["status"] for r in previous["results"]}
            for r in latest["results"]:
                if before.get(r["scenario"]) == "pass" and r["status"] != "pass":
                    regressions.append(r["scenario"])
            lines += [
                "",
                f"compared with {previous['id']}: {previous['summary'].get('pass_rate', 0):.0%} → {s.get('pass_rate', 0):.0%}",
            ]
            lines.append(f"regressions: {', '.join(regressions) if regressions else 'none'}")
        lines += ["", "recent runs:"]
        for run in runs:
            rs = run["summary"]
            lines.append(
                f"  {run['id']} {run['suite']:<10} {rs.get('passed', 0)}/{rs.get('total', 0)} ({rs.get('pass_rate', 0):.0%}) {run['label'] or ''}"
            )
    return {"runs": runs, "regressions": regressions, "text": "\n".join(lines)}
