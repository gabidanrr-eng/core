"""Database restore and portable export/import bundles.

A bundle is a versioned JSON document holding the redacted records of one task (with its
subtasks), one session, or a whole project. Import is additive and conservative:

* existing rows are never overwritten (they are counted as skipped);
* imported work is history, never runnable: non-terminal tasks become ``cancelled`` and
  non-terminal attempts ``interrupted``, and workspace links are dropped;
* imported memory becomes ``stale`` with ``source_type = 'import'`` so it must be re-confirmed
  against this checkout before the context compiler treats it as current.
"""

from __future__ import annotations

import base64
import contextlib
import os
import socket
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coremain.domain.states import TERMINAL, AttemptStatus
from coremain.errors import ConflictError, IntegrityFailure, UsageError
from coremain.paths import CorePaths
from coremain.store.migrate import load_migrations
from coremain.util.jsonutil import dumps, loads
from coremain.version import __version__

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime

BUNDLE_FORMAT = "coremain.bundle"
BUNDLE_VERSION = 1
# Import order respects foreign keys (parents first).
TABLES = (
    "sessions",
    "messages",
    "tasks",
    "task_deps",
    "attempts",
    "evidence",
    "reviews",
    "findings",
    "debug_hypotheses",
    "memory",
    "decisions",
    "artifacts",
)
_AUTOINCREMENT = {"messages": "seq", "memory": "seq"}
_TEXT_TYPES = ("text/", "application/json", "application/x-ndjson")
MAX_EMBED_BYTES = 2_000_000
LIVE_RUNTIME_WINDOW_S = 60.0


# ============================================================================ restore
def _live_runtimes(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, pid, host, heartbeat_at FROM runtimes WHERE status = 'running'"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    live = []
    host = socket.gethostname()
    now = time.time()
    for r in rows:
        if r["host"] == host:
            if r["pid"] == os.getpid():
                continue
            try:
                os.kill(int(r["pid"]), 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                pass
            live.append(dict(r))
        elif now - float(r["heartbeat_at"] or 0) < LIVE_RUNTIME_WINDOW_S:
            live.append(dict(r))
    return live


def _validate_backup(path: Path) -> int:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as exc:
        raise IntegrityFailure(f"{path} is not a readable SQLite database: {exc}") from exc
    try:
        result = conn.execute("PRAGMA integrity_check").fetchall()
        if [r[0] for r in result] != ["ok"]:
            raise IntegrityFailure(
                f"{path} failed integrity_check: {'; '.join(str(r[0]) for r in result[:5])}"
            )
        try:
            version = int(
                conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()[0]
            )
        except sqlite3.Error as exc:
            raise IntegrityFailure(
                f"{path} is not a Core Main database (no schema_migrations table)"
            ) from exc
    except sqlite3.DatabaseError as exc:
        raise IntegrityFailure(f"{path} is not a valid SQLite database: {exc}") from exc
    finally:
        conn.close()
    latest = max(m.version for m in load_migrations())
    if version > latest:
        raise IntegrityFailure(
            f"{path} has schema v{version}, newer than this Core Main (v{latest}); upgrade Core Main first"
        )
    return version


def restore_database(paths: CorePaths, backup: Path) -> dict[str, Any]:
    """Replace the database with ``backup`` after validating it and saving the current one."""
    backup = backup.expanduser().resolve()
    if backup == paths.db_path.resolve():
        raise UsageError("the backup is the live database itself")
    schema = _validate_backup(backup)
    live = _live_runtimes(paths.db_path)
    if live:
        raise ConflictError(
            f"{len(live)} other Core Main runtime(s) are using the database (pids {', '.join(str(r['pid']) for r in live)})",
            hint="close other `core` sessions (TUI, serve, running tasks) and retry",
        )
    paths.backups_dir.mkdir(parents=True, exist_ok=True)
    previous: Path | None = None
    if paths.db_path.exists():
        previous = paths.backups_dir / f"core-pre-restore-{time.strftime('%Y%m%d-%H%M%S')}.db"
        src = sqlite3.connect(str(paths.db_path), timeout=15)
        dst = sqlite3.connect(str(previous))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    tmp = paths.db_path.with_name(f".{paths.db_path.name}.restore-{os.getpid()}")
    src = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    dst = sqlite3.connect(str(tmp))
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")
    finally:
        dst.close()
        src.close()
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"{paths.db_path}{suffix}")
    os.replace(tmp, paths.db_path)
    with contextlib.suppress(OSError):
        os.chmod(paths.db_path, 0o600)
    return {
        "restored_from": str(backup),
        "previous_backup": str(previous) if previous else None,
        "schema": schema,
    }


# ============================================================================ export
def _rows(rt: CoreRuntime, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in rt.db.query(sql, tuple(params))]


def _in(ids: Iterable[str]) -> tuple[str, list[str]]:
    values = sorted(set(ids))
    return ",".join("?" * len(values)) or "NULL", values


def _task_tree(rt: CoreRuntime, root_ids: list[str]) -> list[str]:
    seen: list[str] = []
    frontier = list(root_ids)
    while frontier:
        seen.extend(t for t in frontier if t not in seen)
        marks, values = _in(frontier)
        frontier = [
            r["id"] for r in rt.db.query(f"SELECT id FROM tasks WHERE parent_task_id IN ({marks})", values)
        ]
        frontier = [t for t in frontier if t not in seen]
    return seen


def export_bundle(
    rt: CoreRuntime,
    output: Path,
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    whole_project: bool = False,
    include_artifacts: bool = False,
) -> dict[str, Any]:
    project = rt.require_project()
    data: dict[str, list[dict[str, Any]]] = {t: [] for t in TABLES}
    if whole_project:
        data["sessions"] = _rows(
            rt, "SELECT * FROM sessions WHERE project_id = ? ORDER BY created_at", (project.id,)
        )
        task_ids = [r["id"] for r in rt.db.query("SELECT id FROM tasks WHERE project_id = ?", (project.id,))]
        scope_sql, scope_params = "project_id = ?", [project.id]
    elif session_id:
        session = rt.sessions.resolve(session_id, project_id=project.id)
        data["sessions"] = _rows(rt, "SELECT * FROM sessions WHERE id = ?", (session.id,))
        task_ids = [r["id"] for r in rt.db.query("SELECT id FROM tasks WHERE session_id = ?", (session.id,))]
        scope_sql, scope_params = "session_id = ?", [session.id]
    elif task_id:
        task = rt.tasks.resolve(task_id, project_id=project.id)
        task_ids = _task_tree(rt, [task.id])
        if task.session_id:
            data["sessions"] = _rows(rt, "SELECT * FROM sessions WHERE id = ?", (task.session_id,))
        marks, values = _in(task_ids)
        scope_sql, scope_params = f"task_id IN ({marks})", values
    else:
        raise UsageError("choose a session, a task or the whole project to export")
    marks, values = _in(task_ids)
    session_ids = [s["id"] for s in data["sessions"]]
    smarks, svalues = _in(session_ids)
    if task_id:
        data["messages"] = _rows(
            rt, f"SELECT * FROM messages WHERE task_id IN ({marks}) ORDER BY seq", values
        )
    else:
        data["messages"] = _rows(
            rt, f"SELECT * FROM messages WHERE session_id IN ({smarks}) ORDER BY seq", svalues
        )
    data["tasks"] = _rows(rt, f"SELECT * FROM tasks WHERE id IN ({marks}) ORDER BY depth, created_at", values)
    data["task_deps"] = _rows(rt, f"SELECT * FROM task_deps WHERE task_id IN ({marks})", values)
    for table in ("attempts", "evidence", "reviews", "findings", "debug_hypotheses"):
        order = next((c for c in ("created_at", "started_at") if _has_column(rt, table, c)), "rowid")
        data[table] = _rows(rt, f"SELECT * FROM {table} WHERE task_id IN ({marks}) ORDER BY {order}", values)
    data["memory"] = _rows(
        rt,
        f"SELECT * FROM memory WHERE project_id = ? AND {scope_sql} ORDER BY seq",
        [project.id, *scope_params],
    )
    data["decisions"] = _rows(
        rt,
        f"SELECT * FROM decisions WHERE project_id = ? AND {scope_sql} ORDER BY created_at",
        [project.id, *scope_params],
    )
    artifacts = _rows(
        rt, f"SELECT * FROM artifacts WHERE task_id IN ({marks}) AND state != 'expired'", values
    )
    embedded = 0
    for art in artifacts:
        art["content"] = None
        if (
            include_artifacts
            and art["size"] <= MAX_EMBED_BYTES
            and str(art["media_type"]).startswith(_TEXT_TYPES)
        ):
            try:
                art["content"] = rt.redactor.redact(rt.artifacts.read_text(art["id"]))
                art["content_encoding"] = "text"
                embedded += 1
            except Exception as exc:  # noqa: BLE001 - a missing blob must not abort the export
                art["content_error"] = str(exc)
    data["artifacts"] = artifacts
    bundle = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "core_version": __version__,
        "exported_at": rt.clock.now(),
        "scope": {"session_id": session_id, "task_id": task_id, "project": whole_project},
        "project": {"id": project.id, "name": project.name},
        "tables": rt.redactor.redact_obj(data),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(dumps(bundle, indent=2), encoding="utf-8")
    os.replace(tmp, output)
    counts = {t: len(rows) for t, rows in data.items() if rows}
    return {"output": str(output), "counts": counts, "embedded_artifacts": embedded}


def _has_column(rt: CoreRuntime, table: str, column: str) -> bool:
    return any(r["name"] == column for r in rt.db.query(f"PRAGMA table_info({table})"))


# ============================================================================ import
def _columns(conn: sqlite3.Connection, table: str) -> dict[str, bool]:
    """column name → nullable"""
    return {r[1]: not r[3] for r in conn.execute(f"PRAGMA table_info({table})")}


def _exists(conn: sqlite3.Connection, table: str, key: str, value: Any) -> bool:
    return conn.execute(f"SELECT 1 FROM {table} WHERE {key} = ?", (value,)).fetchone() is not None


def import_bundle(rt: CoreRuntime, bundle_path: Path) -> dict[str, Any]:
    project = rt.require_project()
    raw = loads(bundle_path.read_text(encoding="utf-8"), None)
    if not isinstance(raw, dict) or raw.get("format") != BUNDLE_FORMAT:
        raise UsageError(f"{bundle_path} is not a Core Main bundle")
    if int(raw.get("version", 0)) > BUNDLE_VERSION:
        raise UsageError(
            f"bundle version {raw.get('version')} is newer than supported ({BUNDLE_VERSION}); upgrade Core Main"
        )
    tables: dict[str, list[dict[str, Any]]] = raw.get("tables") or {}
    counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    now = rt.clock.now()
    terminal = {s.value for s in TERMINAL}
    attempt_terminal = {
        AttemptStatus.SUCCEEDED,
        AttemptStatus.FAILED,
        AttemptStatus.CANCELLED,
        AttemptStatus.INTERRUPTED,
    }
    with rt.db.tx() as conn:
        for table in TABLES:
            rows = tables.get(table) or []
            if not rows:
                continue
            cols = _columns(conn, table)
            fks = [(r[3], r[2], r[4]) for r in conn.execute(f"PRAGMA foreign_key_list({table})")]
            for original in rows:
                row = {k: v for k, v in original.items() if k in cols and k != _AUTOINCREMENT.get(table)}
                content = original.get("content")
                if table == "task_deps":
                    duplicate = conn.execute(
                        "SELECT 1 FROM task_deps WHERE task_id = ? AND depends_on = ?",
                        (row.get("task_id"), row.get("depends_on")),
                    ).fetchone()
                    if duplicate:
                        skipped[table] = skipped.get(table, 0) + 1
                        continue
                elif "id" in row and _exists(conn, table, "id", row["id"]):
                    skipped[table] = skipped.get(table, 0) + 1
                    continue
                if "project_id" in row:
                    row["project_id"] = project.id
                if table == "tasks":
                    row["workspace_id"] = None
                    row["idempotency_key"] = None
                    if row.get("status") not in terminal:
                        row["status_reason"] = f"imported from bundle (was {row.get('status')})"
                        row["status"] = "cancelled"
                        row["completed_at"] = row.get("completed_at") or now
                if table == "attempts":
                    row["workspace_id"] = None
                    if row.get("status") not in attempt_terminal:
                        row["status"] = AttemptStatus.INTERRUPTED.value
                        row["ended_at"] = row.get("ended_at") or now
                if table == "memory":
                    row["status"] = "stale"
                    row["status_reason"] = "imported from a bundle; re-confirm against this checkout"
                    row["source_type"] = "import"
                if table == "artifacts":
                    if content is None:
                        skipped[table] = skipped.get(table, 0) + 1
                        continue
                    data = (
                        base64.b64decode(content)
                        if original.get("content_encoding") == "base64"
                        else str(content).encode("utf-8")
                    )
                    rt.artifacts.put_bytes(
                        data,
                        kind=row["kind"],
                        project_id=project.id,
                        task_id=row.get("task_id"),
                        attempt_id=row.get("attempt_id"),
                        name=row.get("name"),
                        media_type=row.get("media_type") or "text/plain; charset=utf-8",
                        meta=loads(row.get("meta_json"), {}) | {"imported_from": str(bundle_path)},
                        artifact_id=row["id"],
                    )
                    counts[table] = counts.get(table, 0) + 1
                    continue
                missing_required = False
                for column, ref_table, ref_column in fks:
                    value = row.get(column)
                    if value is None or _exists(conn, ref_table, ref_column, value):
                        continue
                    if cols.get(column, True):
                        row[column] = None
                    else:
                        missing_required = True
                if missing_required:
                    skipped[table] = skipped.get(table, 0) + 1
                    continue
                names = list(row)
                conn.execute(
                    f"INSERT INTO {table}({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                    [row[n] for n in names],
                )
                counts[table] = counts.get(table, 0) + 1
    rt.events.emit(
        "bundle.imported",
        project_id=project.id,
        data={"bundle": str(bundle_path), "counts": counts, "skipped": skipped},
    )
    return {"counts": counts, "skipped": skipped}
