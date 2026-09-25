"""Learning loop: recorded outcomes → patterns → versioned heuristics → validation → activation.

Nothing here changes behaviour on its own. ``analyze`` only *proposes* heuristics, each tied
to the concrete records that motivated it; a heuristic influences routing or prompts only after
it is validated on an evaluation suite (non-inferior to the current strategy, no scenario
regressions) and explicitly activated. Every activation can be rolled back to the previous
version, and all transitions are durable events.

Heuristic kinds:

* ``routing`` — ``{"model", "role"?, "task_kind"?, "adjust"}``: a score adjustment the router
  applies (and explains) for that model.
* ``operational`` — ``{"role"?, "task_kind"?, "text"}``: guidance appended to the role's system
  prompt under "Learned guidance".
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from coremain.errors import ConflictError, NotFoundError, UsageError
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, loads, sha256_hex
from coremain.util.text import one_line

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime

MIN_MODEL_SAMPLES = 5
MIN_TOOL_FAILURES = 3
MIN_RECURRENCE = 2

# Known tool failure classes → actionable guidance for mutating roles.
TOOL_GUIDANCE = {
    "read_required": "Always read a file with read_file before editing, writing or deleting it.",
    "no_match": "Copy edit_file old_string exactly from the most recent read_file output, including indentation.",
    "ambiguous_match": "Make edit_file old_string unique by including surrounding lines, or set replace_all deliberately.",
    "path_outside_workspace": "Use workspace-relative paths; never reference files outside the workspace.",
    "policy_denied": "A command or action was denied by policy; choose an allowed alternative instead of retrying it.",
    "timeout": "Long-running commands time out; run narrower commands (single test files, targeted builds).",
}


def _heuristic_row(rt: CoreRuntime, heuristic_id: str) -> dict[str, Any]:
    row = rt.db.one("SELECT * FROM heuristics WHERE id = ? OR id LIKE ?", (heuristic_id, f"%{heuristic_id}"))
    if row is None:
        raise NotFoundError(f"no heuristic {heuristic_id}", hint="list them with `core learn list`")
    return dict(row)


def _refresh(rt: CoreRuntime) -> None:
    rt.router.heuristics = rt._active_heuristics("routing")


# ================================================================== analysis
def failure_patterns(rt: CoreRuntime) -> list[dict[str, Any]]:
    rows = rt.db.query(
        "SELECT signature, category, error_class, COUNT(*) AS n, SUM(CASE WHEN outcome = 'resolved' THEN 1 ELSE 0 END) AS resolved, "
        "MAX(summary) AS example, GROUP_CONCAT(id) AS ids, GROUP_CONCAT(DISTINCT stage) AS stages FROM failures "
        "GROUP BY signature HAVING COUNT(*) >= ? ORDER BY n DESC LIMIT 50",
        (MIN_RECURRENCE,),
    )
    return [
        {
            "signature": r["signature"],
            "category": r["category"],
            "error_class": r["error_class"],
            "count": r["n"],
            "resolved": r["resolved"],
            "stages": (r["stages"] or "").split(","),
            "example": one_line(r["example"] or "", 200),
            "failure_ids": (r["ids"] or "").split(",")[:20],
        }
        for r in rows
    ]


def tool_patterns(rt: CoreRuntime) -> list[dict[str, Any]]:
    rows = rt.db.query(
        "SELECT tool, error_class, COUNT(*) AS n, MAX(error_message) AS example, GROUP_CONCAT(DISTINCT task_id) AS tasks "
        "FROM tool_calls WHERE status IN ('error', 'denied', 'timeout') AND error_class IS NOT NULL "
        "GROUP BY tool, error_class HAVING COUNT(*) >= ? ORDER BY n DESC LIMIT 30",
        (MIN_TOOL_FAILURES,),
    )
    return [
        {
            "tool": r["tool"],
            "error_class": r["error_class"],
            "count": r["n"],
            "example": one_line(r["example"] or "", 160),
            "tasks": [t for t in (r["tasks"] or "").split(",") if t][:10],
        }
        for r in rows
    ]


def model_patterns(rt: CoreRuntime) -> list[dict[str, Any]]:
    """Per (model, role, task kind) outcome rates versus the role/kind average."""
    rows = rt.db.query(
        "SELECT mc.model_key, mc.role, t.kind, t.id AS task_id, t.status FROM model_calls mc JOIN tasks t ON t.id = mc.task_id "
        "WHERE mc.model_key IS NOT NULL AND mc.role IN ('implementer', 'debugger', 'corrector', 'researcher') "
        "AND t.status IN ('completed', 'failed', 'incomplete') GROUP BY mc.model_key, mc.role, t.id"
    )
    groups: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    baseline: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        ok = r["status"] == "completed"
        groups[(r["model_key"], r["role"], r["kind"])].append(ok)
        baseline[(r["role"], r["kind"])].append(ok)
    out = []
    for (model, role, kind), outcomes in groups.items():
        base = baseline[(role, kind)]
        others = len(base) - len(outcomes)
        if len(outcomes) < MIN_MODEL_SAMPLES or others < MIN_MODEL_SAMPLES:
            continue
        rate = sum(outcomes) / len(outcomes)
        other_rate = (sum(base) - sum(outcomes)) / others
        out.append(
            {
                "model": model,
                "role": role,
                "task_kind": kind,
                "samples": len(outcomes),
                "completion_rate": round(rate, 3),
                "others_rate": round(other_rate, 3),
                "delta": round(rate - other_rate, 3),
            }
        )
    errors = rt.db.query(
        "SELECT model_key, error_class, COUNT(*) AS n, (SELECT COUNT(*) FROM model_calls m2 WHERE m2.model_key = mc.model_key) AS total "
        "FROM model_calls mc WHERE status = 'error' AND model_key IS NOT NULL GROUP BY model_key, error_class"
    )
    for r in errors:
        if r["total"] >= MIN_MODEL_SAMPLES and r["n"] / r["total"] >= 0.3:
            out.append(
                {
                    "model": r["model_key"],
                    "role": None,
                    "task_kind": None,
                    "samples": r["total"],
                    "error_class": r["error_class"],
                    "error_rate": round(r["n"] / r["total"], 3),
                }
            )
    return sorted(out, key=lambda p: -abs(p.get("delta", p.get("error_rate", 0))))


def _proposals(
    rt: CoreRuntime, failures: list[dict[str, Any]], tools: list[dict[str, Any]], models: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for t in tools:
        text = TOOL_GUIDANCE.get(t["error_class"])
        if text is None:
            continue
        proposals.append(
            {
                "name": f"guidance.tool.{t['error_class']}",
                "kind": "operational",
                "content": {"role": None, "task_kind": None, "text": text},
                "rationale": f"{t['tool']} failed {t['count']}x with {t['error_class']} (e.g. {t['example']})",
                "sources": t["tasks"],
            }
        )
    for f in failures:
        if f["category"] in {"verification", "review"} and f["count"] >= MIN_RECURRENCE:
            proposals.append(
                {
                    "name": f"guidance.recurring.{f['signature'][:12]}",
                    "kind": "operational",
                    "content": {
                        "role": None,
                        "task_kind": None,
                        "text": f'A recurring {f["category"]} failure in this project: "{f["example"]}". Check for it explicitly '
                        "and run the relevant verification before submitting.",
                    },
                    "rationale": f"{f['count']} {f['category']} failures share signature {f['signature']}",
                    "sources": f["failure_ids"],
                }
            )
    for m in models:
        if "delta" in m and abs(m["delta"]) >= 0.2:
            adjust = round(max(-0.15, min(0.15, m["delta"] / 2)), 3)
            proposals.append(
                {
                    "name": f"routing.{m['model']}.{m['role']}.{m['task_kind']}",
                    "kind": "routing",
                    "content": {
                        "model": m["model"],
                        "role": m["role"],
                        "task_kind": m["task_kind"],
                        "adjust": adjust,
                    },
                    "rationale": f"{m['model']} as {m['role']} on {m['task_kind']} tasks: {m['completion_rate']:.0%} completion over "
                    f"{m['samples']} tasks vs {m['others_rate']:.0%} for other models",
                    "sources": [],
                }
            )
        elif m.get("error_class") in {"timeout", "server_error", "overloaded", "rate_limited", "network"}:
            proposals.append(
                {
                    "name": f"routing.{m['model']}.reliability",
                    "kind": "routing",
                    "content": {"model": m["model"], "role": None, "task_kind": None, "adjust": -0.1},
                    "rationale": f"{m['model']} failed {m['error_rate']:.0%} of {m['samples']} calls with {m['error_class']}",
                    "sources": [],
                }
            )
    return proposals


def _record_proposal(rt: CoreRuntime, p: dict[str, Any]) -> dict[str, Any] | None:
    content_json = dumps(p["content"])
    existing = rt.db.query(
        "SELECT id, version, content_json, status FROM heuristics WHERE name = ? ORDER BY version DESC",
        (p["name"],),
    )
    if any(r["content_json"] == content_json for r in existing):
        return None
    version = (existing[0]["version"] + 1) if existing else 1
    now = rt.clock.now()
    hid = new_id("heu", now=now)
    rt.db.execute(
        "INSERT INTO heuristics(id, name, version, kind, content_json, rationale, source_failure_ids_json, status, validation_json, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            hid,
            p["name"],
            version,
            p["kind"],
            content_json,
            p["rationale"],
            dumps(p["sources"]),
            "proposed",
            "{}",
            now,
            now,
        ),
    )
    rt.events.emit(
        "heuristic.proposed",
        data={"id": hid, "name": p["name"], "version": version, "rationale": p["rationale"]},
    )
    return {"id": hid, "name": p["name"], "version": version, "kind": p["kind"], "rationale": p["rationale"]}


def analyze(rt: CoreRuntime, *, propose: bool = True) -> dict[str, Any]:
    failures = failure_patterns(rt)
    tools = tool_patterns(rt)
    models = model_patterns(rt)
    candidates = _proposals(rt, failures, tools, models)
    recorded = [r for r in (_record_proposal(rt, p) for p in candidates) if r] if propose else []
    open_failures = rt.db.scalar("SELECT COUNT(*) FROM failures WHERE outcome = 'open'") or 0
    regression_candidates = (
        rt.db.scalar(
            "SELECT COUNT(*) FROM failures WHERE regression_candidate = 1 AND id NOT IN "
            "(SELECT failure_id FROM regression_cases WHERE failure_id IS NOT NULL)"
        )
        or 0
    )
    lines = [
        f"Recorded failures: {open_failures} open; recurring patterns: {len(failures)}; tool failure clusters: {len(tools)}"
    ]
    for f in failures[:10]:
        lines.append(
            f"  • {f['category']}/{f['error_class']} ×{f['count']} ({f['resolved']} resolved): {f['example']}"
        )
    for t in tools[:10]:
        lines.append(f"  • tool {t['tool']} {t['error_class']} ×{t['count']}: {t['example']}")
    for m in models[:10]:
        if "delta" in m:
            lines.append(
                f"  • {m['model']} as {m['role']} on {m['task_kind']}: {m['completion_rate']:.0%} vs {m['others_rate']:.0%} (n={m['samples']})"
            )
        else:
            lines.append(
                f"  • {m['model']}: {m['error_rate']:.0%} {m['error_class']} errors (n={m['samples']})"
            )
    if not (failures or tools or models):
        lines.append("  (not enough recorded outcomes for patterns yet)")
    if propose:
        lines.append(
            f"New heuristic proposals: {len(recorded)}"
            + (" — validate with `core learn validate <id>`" if recorded else "")
        )
        lines += [f"  + {r['id']} {r['name']} v{r['version']}: {r['rationale']}" for r in recorded]
    else:
        lines.append(f"Candidate heuristics (not recorded): {len(candidates)}")
    if regression_candidates:
        lines.append(
            f"{regression_candidates} resolved failure(s) can become regression cases (`core learn regressions`)."
        )
    return {
        "failure_patterns": failures,
        "tool_patterns": tools,
        "model_patterns": models,
        "proposals": recorded,
        "candidates": candidates,
        "text": "\n".join(lines),
    }


# ================================================================ lifecycle
async def validate_heuristic(
    rt: CoreRuntime, heuristic_id: str, *, suite: str = "smoke", scenario_ids: list[str] | None = None
) -> dict[str, Any]:
    """Run ``suite`` with and without the heuristic; validated means non-inferior with no regressions."""
    from coremain.evals.harness import EvalHarness

    h = _heuristic_row(rt, heuristic_id)
    if h["status"] not in {"proposed", "validated"}:
        raise ConflictError(
            f"heuristic {h['id']} is {h['status']}; only proposed or validated heuristics can be validated"
        )
    harness = EvalHarness(rt)
    others = [a for a in harness.active_heuristics() if a["name"] != h["name"]]
    baseline = await harness.run(
        suite=suite,
        scenario_ids=scenario_ids,
        heuristics=others,
        label=f"validate {h['name']} v{h['version']}: baseline",
    )
    candidate = await harness.run(
        suite=suite,
        scenario_ids=scenario_ids,
        heuristics=[*others, h],
        label=f"validate {h['name']} v{h['version']}: candidate",
    )
    before = {r["scenario"]: r["status"] for r in baseline["results"]}
    regressions = [
        r["scenario"]
        for r in candidate["results"]
        if before.get(r["scenario"]) == "pass" and r["status"] != "pass"
    ]
    b_rate, c_rate = baseline["summary"]["pass_rate"], candidate["summary"]["pass_rate"]
    counted = candidate["summary"]["total"]
    if counted == 0:
        status, reason = h["status"], f"suite {suite} ran no scenarios; nothing was validated"
    elif regressions or c_rate < b_rate:
        status = "rejected"
        reason = f"pass rate {b_rate:.0%} → {c_rate:.0%}; regressions: {', '.join(regressions) or 'none'}"
    else:
        status = "validated"
        reason = f"pass rate {b_rate:.0%} → {c_rate:.0%}, no regressions over {counted} scenario(s)"
        if baseline["strategy_detail"]["scripted"]:
            reason += " (scripted suite: checks non-inferiority of the machinery only; use --suite core for live models)"
    validation = {
        "suite": suite,
        "baseline_run": baseline["id"],
        "candidate_run": candidate["id"],
        "baseline_pass_rate": b_rate,
        "candidate_pass_rate": c_rate,
        "regressions": regressions,
        "reason": reason,
    }
    rt.db.execute(
        "UPDATE heuristics SET status = ?, validation_json = ?, updated_at = ? WHERE id = ?",
        (status, dumps(validation), rt.clock.now(), h["id"]),
    )
    rt.events.emit(
        f"heuristic.{status}", data={"id": h["id"], "name": h["name"], "version": h["version"], **validation}
    )
    return {"id": h["id"], "status": status, "reason": reason, "validation": validation}


def set_heuristic_status(
    rt: CoreRuntime, heuristic_id: str, status: str, *, force: bool = False
) -> dict[str, Any]:
    h = _heuristic_row(rt, heuristic_id)
    now = rt.clock.now()
    validation = loads(h["validation_json"], {})
    if status == "active":
        if h["status"] == "active":
            return h
        if h["status"] != "validated" and not force:
            raise ConflictError(
                f"heuristic {h['id']} is {h['status']}; validate it first or pass --force",
                hint=f"core learn validate {h['id']}",
            )
        if h["status"] == "rejected" and not force:
            raise ConflictError(f"heuristic {h['id']} was rejected")
        with rt.db.tx() as conn:
            for prev in conn.execute(
                "SELECT id, validation_json FROM heuristics WHERE name = ? AND status = 'active'",
                (h["name"],),
            ):
                pv = loads(prev["validation_json"], {}) | {"retired_reason": f"superseded by v{h['version']}"}
                conn.execute(
                    "UPDATE heuristics SET status = 'retired', validation_json = ?, updated_at = ? WHERE id = ?",
                    (dumps(pv), now, prev["id"]),
                )
            if force and h["status"] != "validated":
                validation["forced"] = {"at": now, "from_status": h["status"]}
            conn.execute(
                "UPDATE heuristics SET status = 'active', validation_json = ?, updated_at = ? WHERE id = ?",
                (dumps(validation), now, h["id"]),
            )
    elif status == "rejected":
        rt.db.execute(
            "UPDATE heuristics SET status = 'rejected', updated_at = ? WHERE id = ?", (now, h["id"])
        )
    else:
        raise UsageError(f"unsupported heuristic status {status}")
    rt.events.emit(
        f"heuristic.{status}",
        data={"id": h["id"], "name": h["name"], "version": h["version"], "forced": force},
    )
    _refresh(rt)
    return _heuristic_row(rt, h["id"])


def rollback(rt: CoreRuntime, name: str) -> dict[str, Any]:
    active = rt.db.one("SELECT * FROM heuristics WHERE name = ? AND status = 'active'", (name,))
    if active is None:
        raise NotFoundError(f"heuristic {name} has no active version")
    now = rt.clock.now()
    previous = None
    for row in rt.db.query(
        "SELECT * FROM heuristics WHERE name = ? AND status = 'retired' AND version < ? ORDER BY version DESC",
        (name, active["version"]),
    ):
        if str(loads(row["validation_json"], {}).get("retired_reason", "")).startswith("superseded"):
            previous = row
            break
    with rt.db.tx() as conn:
        av = loads(active["validation_json"], {}) | {"retired_reason": "rolled back"}
        conn.execute(
            "UPDATE heuristics SET status = 'retired', validation_json = ?, updated_at = ? WHERE id = ?",
            (dumps(av), now, active["id"]),
        )
        if previous is not None:
            conn.execute(
                "UPDATE heuristics SET status = 'active', updated_at = ? WHERE id = ?", (now, previous["id"])
            )
    rt.events.emit(
        "heuristic.rolled_back",
        data={
            "name": name,
            "from_version": active["version"],
            "to_version": previous["version"] if previous else None,
        },
    )
    _refresh(rt)
    return {
        "name": name,
        "rolled_back_version": active["version"],
        "active_version": previous["version"] if previous else None,
    }


def guidance_for(rt: CoreRuntime, role: str, task_kind: str | None) -> list[str]:
    return [
        f"{h['text']} [{h['name']}]"
        for h in rt._active_heuristics("operational")
        if h.get("text") and h.get("role") in (None, role) and h.get("task_kind") in (None, task_kind)
    ]


# =============================================================== regressions
def create_regression_case(rt: CoreRuntime, failure_id: str) -> dict[str, Any]:
    """Turn a failure into a replayable scenario pinned to the commit the task started from.

    The repository state is kept reachable with a hidden ref (``refs/core/regressions/<id>``) so
    workspace garbage collection cannot lose it; no branch is created.
    """
    row = rt.db.one("SELECT * FROM failures WHERE id = ? OR id LIKE ?", (failure_id, f"%{failure_id}"))
    if row is None:
        raise NotFoundError(f"no failure {failure_id}")
    failure = dict(row)
    if rt.db.one("SELECT id FROM regression_cases WHERE failure_id = ?", (failure["id"],)):
        raise ConflictError(f"failure {failure['id']} already has a regression case")
    if not failure["task_id"]:
        raise UsageError("this failure is not attached to a task, so it cannot be replayed")
    task = rt.tasks.get(failure["task_id"])
    project = rt.projects.get(task.project_id)
    ws = (
        rt.db.one("SELECT base_ref FROM workspaces WHERE id = ?", (task.workspace_id,))
        if task.workspace_id
        else None
    )
    commit = ws["base_ref"] if ws and ws["base_ref"] else None
    if commit is None:
        raise UsageError("the task has no recorded base commit (it did not run in an isolated git workspace)")
    case_id = new_id("reg", now=rt.clock.now())
    ref = f"refs/core/regressions/{case_id}"
    result = subprocess.run(
        ["git", "-C", project.root_path, "update-ref", ref, commit],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ConflictError(f"cannot pin commit {commit[:12]}: {result.stderr.strip()}")
    contract = {k: v for k, v in task.contract.items() if k != "requirements"}
    checks: list[dict[str, Any]] = [{"status": "completed"}]
    if task.evidence_level:
        checks.append({"min_level": task.evidence_level if task.evidence_level != "none" else "weak"})
    if failure["category"] == "verification":
        checks.append({"evidence": {"kind": "tests", "status": "pass"}})
    scenario = {
        "id": f"regression-{case_id[-10:]}",
        "family": "regression",
        "title": one_line(f"{failure['category']}: {failure['summary']}", 120),
        "request": task.description,
        "mode": task.mode,
        "contract": contract,
        "repo": {"path": project.root_path, "ref": ref, "commit": commit},
        "checks": checks,
        "suites": ["regression"],
    }
    rt.db.execute(
        "INSERT INTO regression_cases(id, failure_id, signature, title, scenario_json, status, created_at) VALUES (?,?,?,?,?,?,?)",
        (
            case_id,
            failure["id"],
            failure["signature"],
            scenario["title"],
            dumps(scenario),
            "active",
            rt.clock.now(),
        ),
    )
    rt.events.emit(
        "regression.created",
        project_id=project.id,
        task_id=task.id,
        data={"id": case_id, "failure_id": failure["id"], "commit": commit},
    )
    return {"id": case_id, "scenario": scenario, "fingerprint": sha256_hex(dumps(scenario))[:12]}
