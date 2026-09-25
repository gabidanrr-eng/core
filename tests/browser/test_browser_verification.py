"""Contract browser checks are executed by the verifier and gate completion like tests do."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import Harness, approve_review, make_repo, turn

pytestmark = pytest.mark.browser

PAGE = """<!doctype html><html><head><title>Settings</title></head>
<body><h1>Settings</h1><button id="save">Sav changes</button></body></html>
"""
CHECKS = [
    {
        "name": "save button label",
        "url": "index.html",
        "selector": "#save",
        "expect_text": "Save changes",
        "expect_no_console_errors": True,
        "screenshot": True,
    }
]


def _script(fixed: str) -> dict:
    impl = [
        turn(("read_file", {"path": "index.html"})),
        turn(("edit_file", {"path": "index.html", "old_string": "Sav changes", "new_string": fixed})),
        turn(("submit_result", {"summary": "Fixed the save button label", "files_changed": ["index.html"]})),
    ]
    return {
        "name": "ui",
        "roles": {"implementer": impl, "corrector": [impl[-1]], "reviewer": approve_review()},
    }


async def test_browser_check_passes_and_gates_completion(harness: Harness, tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "site", {"index.html": PAGE})
    harness.scripted(_script("Save changes"), test_command=None)
    async with harness.runtime(repo) as rt:
        task = await rt.submit(
            "Fix the typo in the save button label", mode="direct", contract={"browser_checks": CHECKS}
        )
        final = await rt.run_task(task.id)
        assert final.status == "completed", final.status_reason
        browser = [e for e in rt.evidence.for_task(task.id) if e.kind == "browser"]
        assert browser and browser[-1].status == "pass" and browser[-1].trust == "observed"
        assert browser[-1].artifact_id is not None
        assert rt.artifacts.read_bytes(browser[-1].artifact_id)[:8] == b"\x89PNG\r\n\x1a\n"
    assert "Save changes" in (repo / "index.html").read_text()


async def test_failing_browser_check_blocks_completion(harness: Harness, tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "site", {"index.html": PAGE})
    harness.scripted(_script("Save chnages"), test_command=None)
    async with harness.runtime(repo) as rt:
        task = await rt.submit(
            "Fix the typo in the save button label", mode="direct", contract={"browser_checks": CHECKS}
        )
        final = await rt.run_task(task.id)
        assert final.status != "completed"
        browser = [e for e in rt.evidence.for_task(task.id) if e.kind == "browser"]
        assert browser and all(e.status == "fail" for e in browser)
    assert "Sav changes" in (repo / "index.html").read_text(), "unverified UI work must not be applied"
