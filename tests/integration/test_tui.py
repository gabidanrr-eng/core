"""The Textual UI driven headlessly through Pilot against a real runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import DataTable, Input, Static

from coremain.paths import resolve_core_paths
from coremain.runtime.app import CoreRuntime
from coremain.tui.app import CoreApp
from coremain.tui.screens import ApprovalScreen, HelpScreen
from coremain.tui.widgets import Conversation
from tests.helpers import FIX_CALC_TURNS, Harness, approve_review, turn


async def _wait(pilot, predicate, timeout: float = 60) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await pilot.pause(0.1)


def _runtime(harness: Harness, repo: Path) -> CoreRuntime:
    return CoreRuntime.open(project_root=repo, paths=resolve_core_paths(harness.env), env=harness.env, mode="tui")


async def _type(pilot, text: str) -> None:
    pilot.app.query_one("#input", Input).value = text
    await pilot.press("enter")


async def test_tui_runs_a_task_end_to_end(harness: Harness, calc_repo: Path) -> None:
    harness.scripted({"name": "fix", "roles": {"debugger": FIX_CALC_TURNS, "implementer": FIX_CALC_TURNS, "reviewer": approve_review()}})
    rt = _runtime(harness, calc_repo)
    app = CoreApp(runtime=rt)
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            conv = app.query_one("#conversation", Conversation)
            assert "Welcome to Core Main" in str(conv.children[0].render())
            await _type(pilot, "The test_add test is failing; fix the add function in calc.py")
            await _wait(pilot, lambda: bool(conv.blocks))
            block = next(iter(conv.blocks.values()))
            await _wait(pilot, lambda: block.status == "completed", timeout=90)
            await _wait(pilot, lambda: block.done)
            report = block.query_one(".task-report").source  # type: ignore[attr-defined]
            assert "Task completed" in report and "M calc.py" in report
            app.refresh_panels()
            details = str(app.query_one("#task-details", Static).render())
            assert "completed" in details and "strong" in details
            assert app.query_one("#evidence", DataTable).row_count >= 3
            await pilot.press("f4")
            await pilot.pause(0.5)
            assert "return a + b" in str(app.query_one("#diff-view", Static).render()) or (calc_repo / "calc.py").read_text().count("a + b")
            await _type(pilot, "/mode plan")
            assert app.mode_override == "plan"
            await _type(pilot, "/mode nonsense")
            assert app.mode_override == "plan"
            await _type(pilot, "/model auto")
            assert app.model_override is None
            await pilot.press("f1")
            await pilot.pause(0.2)
            assert isinstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await _type(pilot, "/new Second session")
            await pilot.pause(0.3)
            assert rt.sessions.get(app.session_id).title == "Second session"
            await pilot.press("ctrl+q")
    finally:
        await rt.close()
    assert (calc_repo / "calc.py").read_text().strip().endswith("return a + b")


async def test_tui_approval_modal_gates_a_risky_command(harness: Harness, calc_repo: Path) -> None:
    risky = [
        turn(("run_command", {"command": "git push origin main"})),
        *FIX_CALC_TURNS,
    ]
    harness.scripted({"name": "risky", "roles": {"implementer": risky, "reviewer": approve_review()}})
    rt = _runtime(harness, calc_repo)
    app = CoreApp(runtime=rt)
    try:
        async with app.run_test(size=(140, 44)) as pilot:
            await _type(pilot, "Fix add() in calc.py so it adds")
            conv = app.query_one("#conversation", Conversation)
            await _wait(pilot, lambda: bool(conv.blocks))
            block = next(iter(conv.blocks.values()))
            # Either the policy asks (modal) or denies outright; a push must never just run.
            await _wait(pilot, lambda: isinstance(app.screen, ApprovalScreen) or any("git push" in line for line in block._lines), timeout=60)
            if isinstance(app.screen, ApprovalScreen):
                await pilot.press("n")
            await _wait(pilot, lambda: block.status in {"completed", "failed", "incomplete"}, timeout=90)
            calls = rt.db.query("SELECT status FROM tool_calls WHERE tool = 'run_command'")
            assert calls and all(c["status"] in {"denied", "error"} for c in calls)
            await pilot.press("ctrl+q")
    finally:
        await rt.close()
