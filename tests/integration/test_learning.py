"""Learning loop: patterns from real runs → proposals → validation → activation → rollback,
and resolved failures turned into regression scenarios replayed from a pinned commit."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from coremain.agent.prompts import system_prompt
from coremain.errors import ConflictError
from coremain.evals.harness import EvalHarness
from coremain.learning.analysis import (
    analyze,
    create_regression_case,
    guidance_for,
    rollback,
    set_heuristic_status,
    validate_heuristic,
)
from tests.helpers import FIX_CALC_TURNS, Harness, approve_review, turn

# An implementer that first edits blind (rejected with read_required), then recovers.
BLIND_THEN_READ = [
    turn(("edit_file", {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"})),
    *FIX_CALC_TURNS,
]


async def _runs(harness: Harness, repo: Path, n: int) -> list[str]:
    ids = []
    async with harness.runtime(repo) as rt:
        for _ in range(n):
            task = await rt.submit("Fix add() in calc.py so it adds", mode="direct")
            final = await rt.run_task(task.id)
            assert final.status == "completed", final.status_reason
            subprocess.run(["git", "checkout", "-q", "--", "calc.py"], cwd=repo, check=True)
            ids.append(task.id)
    return ids


async def test_tool_failure_pattern_becomes_validated_active_guidance(
    harness: Harness, calc_repo: Path
) -> None:
    harness.scripted(
        {"name": "blind", "roles": {"implementer": BLIND_THEN_READ, "reviewer": approve_review()}}
    )
    await _runs(harness, calc_repo, 3)
    async with harness.runtime(calc_repo) as rt:
        result = analyze(rt, propose=True)
        assert any(t["error_class"] == "read_required" and t["count"] >= 3 for t in result["tool_patterns"])
        proposal = next(p for p in result["proposals"] if p["name"] == "guidance.tool.read_required")
        assert proposal["version"] == 1
        assert not analyze(rt, propose=True)["proposals"], "identical proposals must not be recorded twice"

        with pytest.raises(ConflictError):
            set_heuristic_status(rt, proposal["id"], "active")  # not validated yet
        validation = await validate_heuristic(
            rt, proposal["id"], suite="smoke", scenario_ids=["py-backend-pagination"]
        )
        assert validation["status"] == "validated", validation["reason"]
        runs = rt.db.query("SELECT label FROM eval_runs ORDER BY started_at")
        assert [r["label"].rsplit(": ", 1)[-1] for r in runs] == ["baseline", "candidate"]

        row = set_heuristic_status(rt, proposal["id"], "active")
        assert row["status"] == "active"
        lines = guidance_for(rt, "implementer", "bugfix")
        assert any("read_file before editing" in g for g in lines)
        prompt = system_prompt("implementer", output_tool="submit_result", guidance=lines)
        assert "Learned guidance" in prompt and "guidance.tool.read_required" in prompt

        # A new version supersedes v1; rolling it back restores v1.
        now = rt.clock.now()
        rt.db.execute(
            "INSERT INTO heuristics(id, name, version, kind, content_json, rationale, source_failure_ids_json, status, "
            "validation_json, created_at, updated_at) VALUES ('heu_v2', 'guidance.tool.read_required', 2, 'operational', "
            "'{\"text\": \"v2 text\"}', 'manual', '[]', 'proposed', '{}', ?, ?)",
            (now, now),
        )
        set_heuristic_status(rt, "heu_v2", "active", force=True)
        statuses = {r["version"]: r["status"] for r in rt.db.query("SELECT version, status FROM heuristics")}
        assert statuses == {1: "retired", 2: "active"}
        assert any("v2 text" in g for g in guidance_for(rt, "implementer", None))
        info = rollback(rt, "guidance.tool.read_required")
        assert info == {"name": "guidance.tool.read_required", "rolled_back_version": 2, "active_version": 1}
        assert not any("v2 text" in g for g in guidance_for(rt, "implementer", None))
        kinds = {e.kind for e in rt.events.since(0, limit=5000)}
        assert {
            "heuristic.proposed",
            "heuristic.validated",
            "heuristic.active",
            "heuristic.rolled_back",
        } <= kinds


async def test_regression_case_replays_pinned_commit(harness: Harness, calc_repo: Path) -> None:
    # First attempt: the model changes nothing, the task fails and a failure is recorded.
    lying = [turn(("submit_result", {"summary": "done", "files_changed": ["calc.py"]}))]
    harness.scripted(
        {"name": "liar", "roles": {"debugger": lying, "implementer": lying, "reviewer": approve_review()}}
    )
    async with harness.runtime(calc_repo) as rt:
        task = await rt.submit("The test_add test is failing; fix the add function in calc.py")
        assert (await rt.run_task(task.id)).status == "failed"
        failure = rt.db.one("SELECT id FROM failures WHERE task_id = ?", (task.id,))
        assert failure is not None, "a failed task must leave a durable failure record"
        case = create_regression_case(rt, failure["id"])
        commit = case["scenario"]["repo"]["commit"]
    # The pinned ref keeps the commit reachable in the user's repository.
    pinned = subprocess.run(
        ["git", "rev-parse", case["scenario"]["repo"]["ref"]], cwd=calc_repo, capture_output=True, text=True
    )
    assert pinned.stdout.strip() == commit
    # Later, with a model that can fix it, the regression suite replays that exact state.
    harness.scripted(
        {
            "name": "fix",
            "roles": {
                "debugger": FIX_CALC_TURNS,
                "implementer": FIX_CALC_TURNS,
                "reviewer": approve_review(),
            },
        }
    )
    async with harness.runtime(calc_repo) as rt:
        run = await EvalHarness(rt).run(suite="regression")
        assert run["summary"]["total"] == 1
        assert run["results"][0]["status"] == "pass", run["results"][0]
