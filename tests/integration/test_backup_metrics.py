"""Backup/restore, export/import bundles and metrics, exercised against a real completed task."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from coremain.errors import ConflictError, IntegrityFailure, UsageError
from coremain.observe.metrics import collect, percentile
from coremain.store.backup import export_bundle, import_bundle, restore_database
from tests.helpers import CALC_REPO, FIX_CALC_TURNS, Harness, approve_review, make_repo


async def _completed_task(harness: Harness, repo: Path) -> str:
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
    async with harness.runtime(repo) as rt:
        task = await rt.submit("The test_add test is failing; fix the add function in calc.py")
        final = await rt.run_task(task.id)
        assert final.status == "completed"
        rt.memory.add(
            content="Tests run with pytest",
            kind="fact",
            scope="operational",
            source_type="user",
            project_id=rt.require_project().id,
        )
        return task.id


async def test_export_import_roundtrip_is_redacted_and_conservative(
    harness: Harness, calc_repo: Path, tmp_path: Path
) -> None:
    task_id = await _completed_task(harness, calc_repo)
    bundle = tmp_path / "bundle.json"
    async with harness.runtime(calc_repo) as rt:
        info = export_bundle(rt, bundle, whole_project=True, include_artifacts=True)
        assert info["counts"]["tasks"] >= 1 and info["counts"]["evidence"] >= 3
        with pytest.raises(UsageError):
            export_bundle(rt, tmp_path / "x.json")
    doc = json.loads(bundle.read_text())
    assert doc["format"] == "coremain.bundle" and doc["version"] == 1

    other = Harness(tmp_path / "other")
    other_repo = make_repo(tmp_path / "other-proj", CALC_REPO)
    async with other.runtime(other_repo) as rt2:
        result = import_bundle(rt2, bundle)
        assert result["counts"]["tasks"] >= 1
        imported = rt2.tasks.get(task_id)
        assert imported.project_id == rt2.require_project().id
        assert imported.status == "completed"
        assert imported.workspace_id is None
        mem = rt2.db.one("SELECT status, source_type FROM memory WHERE content = 'Tests run with pytest'")
        assert mem is not None and (mem["status"], mem["source_type"]) == ("stale", "import")
        # Evidence rows keep resolving to their embedded artifacts.
        linked = rt2.db.query(
            "SELECT artifact_id FROM evidence WHERE task_id = ? AND artifact_id IS NOT NULL", (task_id,)
        )
        for row in linked:
            rt2.artifacts.read_bytes(row["artifact_id"])
        again = import_bundle(rt2, bundle)
        assert not again["counts"], "re-import must not duplicate anything"
        assert again["skipped"]["tasks"] >= 1


async def test_import_neutralizes_unfinished_work(harness: Harness, calc_repo: Path, tmp_path: Path) -> None:
    await _completed_task(harness, calc_repo)
    bundle = tmp_path / "b.json"
    async with harness.runtime(calc_repo) as rt:
        export_bundle(rt, bundle, whole_project=True)
    doc = json.loads(bundle.read_text())
    doc["tables"]["tasks"][0]["status"] = "running"
    doc["tables"]["attempts"][0]["status"] = "running"
    bundle.write_text(json.dumps(doc))
    other = Harness(tmp_path / "o")
    async with other.runtime(make_repo(tmp_path / "o-proj", CALC_REPO)) as rt2:
        import_bundle(rt2, bundle)
        task = rt2.tasks.get(doc["tables"]["tasks"][0]["id"])
        assert task.status == "cancelled" and "imported" in (task.status_reason or "")
        attempt = rt2.tasks.attempt(doc["tables"]["attempts"][0]["id"])
        assert attempt.status == "interrupted"


async def test_restore_replaces_database_and_keeps_previous(
    harness: Harness, calc_repo: Path, tmp_path: Path
) -> None:
    await _completed_task(harness, calc_repo)
    snapshot = tmp_path / "snap.db"
    async with harness.runtime(calc_repo) as rt:
        rt.db.backup_to(snapshot)
        before = rt.db.scalar("SELECT COUNT(*) FROM memory")
        assert before >= 1
        rt.db.execute("DELETE FROM memory")
    info = restore_database(harness.paths, snapshot)
    assert Path(info["previous_backup"]).exists()
    conn = sqlite3.connect(harness.paths.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0] == before
    finally:
        conn.close()
    prev = sqlite3.connect(info["previous_backup"])
    try:
        assert prev.execute("SELECT COUNT(*) FROM memory").fetchone()[0] == 0
    finally:
        prev.close()


async def test_restore_refuses_invalid_backup_and_live_runtimes(
    harness: Harness, calc_repo: Path, tmp_path: Path
) -> None:
    await _completed_task(harness, calc_repo)
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"not a database at all" * 100)
    with pytest.raises(IntegrityFailure):
        restore_database(harness.paths, garbage)
    snapshot = tmp_path / "snap.db"
    async with harness.runtime(calc_repo) as rt:
        rt.db.backup_to(snapshot)
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        conn = sqlite3.connect(harness.paths.db_path)
        conn.execute(
            "INSERT INTO runtimes(id, pid, host, version, mode, started_at, heartbeat_at, status) "
            "VALUES ('rt_other', ?, ?, 'x', 'tui', 0, 0, 'running')",
            (sleeper.pid, __import__("socket").gethostname()),
        )
        conn.commit()
        conn.close()
        with pytest.raises(ConflictError):
            restore_database(harness.paths, snapshot)
    finally:
        sleeper.kill()
        sleeper.wait()


async def test_metrics_reflect_recorded_calls(harness: Harness, calc_repo: Path) -> None:
    await _completed_task(harness, calc_repo)
    async with harness.runtime(calc_repo) as rt:
        data = collect(rt, days=1)
    assert data["tasks"]["total"] == 1
    assert data["tasks"]["by_kind"][0]["completion_rate"] == 1.0
    assert sum(m["calls"] for m in data["models"]) >= 3
    assert {t["tool"] for t in data["tools"]} >= {"read_file", "edit_file", "run_tests"}
    assert "Model calls:" in data["text"] and "scripted" in data["text"]


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
