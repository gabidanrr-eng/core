"""End to end: a failing test is reproduced, fixed in an isolated workspace, verified by really
running pytest, independently reviewed, gated on evidence and applied to the canonical checkout."""

from __future__ import annotations

from pathlib import Path

from coremain.domain.states import TaskStatus
from tests.helpers import FIX_CALC_TURNS, Harness, approve_review, git


async def test_debug_task_fixes_bug_with_real_evidence(harness: Harness, calc_repo: Path) -> None:
    harness.scripted(
        {
            "name": "fix",
            "roles": {
                "implementer": FIX_CALC_TURNS,
                "debugger": FIX_CALC_TURNS,
                "reviewer": approve_review(),
            },
        }
    )
    async with harness.runtime(calc_repo) as rt:
        task = await rt.submit("The test_add test is failing; fix the add function in calc.py")
        assert task.kind == "bugfix"
        assert task.decision["workflow"] == "debug"
        assert any(r["kind"] == "tests" for r in task.contract["requirements"])

        final = await rt.run_task(task.id)

        assert final.status == TaskStatus.COMPLETED, final.status_reason
        assert final.evidence_level == "strong"
        assert (calc_repo / "calc.py").read_text().strip().endswith("return a + b")
        evidence = rt.evidence.for_task(task.id)
        by_kind = {(e.kind, e.status) for e in evidence}
        # The failing test must have been reproduced before the fix, and pass after it.
        assert ("repro", "pass") in by_kind
        assert ("tests", "pass") in by_kind
        assert all(e.trust != "claimed" for e in evidence if e.kind == "tests")
        report = rt.sessions.messages(final.session_id)[-1].content
        assert "completed" in report.lower()
        assert "M calc.py" in report
        events = rt.events.for_task(task.id)
        assert {"task.created", "mode.selected"} <= {ev.kind for ev in events}
        transitions = [ev.data["to"] for ev in events if ev.kind == "task.transition"]
        assert transitions[-1] == "completed"
        # The reproduction ran the configured command, not an auto-detected one.
        repro = next(e for e in evidence if e.kind == "repro")
        assert repro.command is not None and "no:cacheprovider" in repro.command
    # The canonical checkout received exactly the agent's change and nothing else.
    assert git(calc_repo, "status", "--porcelain").strip() == "M calc.py"


async def test_claimed_success_without_evidence_is_not_completed(harness: Harness, calc_repo: Path) -> None:
    """A model that claims success but changes nothing must fail closed, and the report must not
    repeat its claim as fact."""
    lying = [
        {
            "tool_calls": [
                {
                    "name": "submit_result",
                    "arguments": {
                        "summary": "All tests pass now, fixed",
                        "files_changed": ["calc.py"],
                        "confidence": "high",
                    },
                }
            ]
        }
    ]
    harness.scripted(
        {"name": "liar", "roles": {"implementer": lying, "debugger": lying, "reviewer": approve_review()}}
    )
    async with harness.runtime(calc_repo) as rt:
        task = await rt.submit("The test_add test is failing; fix the add function in calc.py")
        final = await rt.run_task(task.id)
        assert final.status == TaskStatus.FAILED
        assert (calc_repo / "calc.py").read_text().strip().endswith("return a - b")
        evidence = rt.evidence.for_task(task.id)
        gate = next(e for e in evidence if e.kind == "gate")
        assert gate.status == "fail"
        claim = next(e for e in evidence if e.kind == "model_claim")
        assert claim.trust == "claimed"
        report = rt.sessions.messages(final.session_id)[-1].content
        assert "not confirmed by evidence" in report
        assert "no modifications" in report
        assert "core task apply" not in report
        assert final.result_summary is not None and final.result_summary.startswith("unverified:")
