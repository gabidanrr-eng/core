"""Content-addressed artifact store.

Blobs live at ``artifacts/<sha[:2]>/<sha>`` and are written atomically (temp file, fsync,
rename), so a crash can leave only an orphaned temp file, never a half-written blob. Rows in
the ``artifacts`` table give blobs ownership (project/task/attempt), kind and lifecycle state;
reads verify the hash and refuse cross-project access when a project is specified.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coremain.errors import IntegrityFailure, NotFoundError
from coremain.security.redact import Redactor
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, loads, sha256_hex


@dataclass
class Artifact:
    id: str
    project_id: str | None
    task_id: str | None
    attempt_id: str | None
    kind: str
    name: str | None
    sha256: str
    size: int
    media_type: str
    state: str
    meta: dict[str, Any]
    created_at: float


@dataclass
class GcReport:
    expired_rows: int = 0
    removed_blobs: int = 0
    removed_temp: int = 0
    freed_bytes: int = 0
    protected: int = 0


class ArtifactStore:
    def __init__(self, db: Database, root: Path, redactor: Redactor, clock: Clock):
        self.db = db
        self.root = root
        self.redactor = redactor
        self.clock = clock
        (root / "tmp").mkdir(parents=True, exist_ok=True)

    def blob_path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha

    def _write_blob(self, data: bytes) -> str:
        sha = sha256_hex(data)
        path = self.blob_path(sha)
        if path.exists() and path.stat().st_size == len(data):
            return sha
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "tmp" / f"{sha}.{os.getpid()}.{time.monotonic_ns()}"
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return sha

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        project_id: str | None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        name: str | None = None,
        media_type: str = "application/octet-stream",
        meta: dict[str, Any] | None = None,
        state: str = "live",
        artifact_id: str | None = None,
    ) -> Artifact:
        """``artifact_id`` preserves an existing identity (bundle import); new ids are generated otherwise."""
        sha = self._write_blob(data)
        now = self.clock.now()
        artifact_id = artifact_id or new_id("art", now=now)
        self.db.execute(
            "INSERT INTO artifacts(id, project_id, task_id, attempt_id, kind, name, sha256, size, media_type, state, meta_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                artifact_id,
                project_id,
                task_id,
                attempt_id,
                kind,
                name,
                sha,
                len(data),
                media_type,
                state,
                dumps(meta or {}),
                now,
            ),
        )
        return Artifact(
            artifact_id,
            project_id,
            task_id,
            attempt_id,
            kind,
            name,
            sha,
            len(data),
            media_type,
            state,
            meta or {},
            now,
        )

    def put_text(
        self, text: str, *, redact: bool = True, media_type: str = "text/plain; charset=utf-8", **kw: Any
    ) -> Artifact:
        if redact:
            text = self.redactor.redact(text)
        return self.put_bytes(text.encode("utf-8"), media_type=media_type, **kw)

    def put_json(self, obj: Any, *, redact: bool = True, **kw: Any) -> Artifact:
        payload = self.redactor.redact_obj(obj) if redact else obj
        return self.put_bytes(dumps(payload).encode("utf-8"), media_type="application/json", **kw)

    def get(self, artifact_id: str) -> Artifact:
        r = self.db.one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        if r is None:
            raise NotFoundError(f"artifact {artifact_id} not found")
        return Artifact(
            r["id"],
            r["project_id"],
            r["task_id"],
            r["attempt_id"],
            r["kind"],
            r["name"],
            r["sha256"],
            r["size"],
            r["media_type"],
            r["state"],
            loads(r["meta_json"], {}),
            r["created_at"],
        )

    def read_bytes(self, artifact_id: str, *, project_id: str | None = None, verify: bool = True) -> bytes:
        art = self.get(artifact_id)
        if project_id is not None and art.project_id not in (None, project_id):
            raise IntegrityFailure(f"artifact {artifact_id} belongs to a different project")
        path = self.blob_path(art.sha256)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise IntegrityFailure(
                f"artifact blob missing for {artifact_id}", hint="Run `core doctor`."
            ) from exc
        if verify and sha256_hex(data) != art.sha256:
            raise IntegrityFailure(f"artifact {artifact_id} failed hash verification (corrupted blob)")
        return data

    def read_text(self, artifact_id: str, **kw: Any) -> str:
        return self.read_bytes(artifact_id, **kw).decode("utf-8", errors="replace")

    def read_json(self, artifact_id: str, **kw: Any) -> Any:
        return loads(self.read_bytes(artifact_id, **kw))

    def for_task(self, task_id: str, kind: str | None = None) -> list[Artifact]:
        sql = (
            "SELECT id FROM artifacts WHERE task_id = ?"
            + (" AND kind = ?" if kind else "")
            + " ORDER BY created_at"
        )
        params = (task_id, kind) if kind else (task_id,)
        return [self.get(r["id"]) for r in self.db.query(sql, params)]

    def protect(self, artifact_id: str) -> None:
        self.db.execute("UPDATE artifacts SET state = 'protected' WHERE id = ?", (artifact_id,))

    def verify(self, *, limit: int | None = None) -> list[str]:
        problems: list[str] = []
        sql = "SELECT id, sha256, size FROM artifacts" + (
            f" ORDER BY created_at DESC LIMIT {int(limit)}" if limit else ""
        )
        for r in self.db.query(sql):
            path = self.blob_path(r["sha256"])
            if not path.exists():
                problems.append(f"{r['id']}: blob missing")
            elif path.stat().st_size != r["size"]:
                problems.append(f"{r['id']}: size mismatch (possible truncation)")
        return problems

    def gc(self, *, max_age_s: float, keep_task_ids: set[str], dry_run: bool = False) -> GcReport:
        """Expire unprotected artifacts of finished work older than ``max_age_s`` and delete
        unreferenced blobs. Artifacts of tasks in ``keep_task_ids`` (active or recoverable
        work) are never expired."""
        report = GcReport()
        cutoff = self.clock.now() - max_age_s
        rows = self.db.query(
            "SELECT id, task_id, state FROM artifacts WHERE state = 'live' AND created_at < ?", (cutoff,)
        )
        expire: list[str] = []
        for r in rows:
            if r["task_id"] and r["task_id"] in keep_task_ids:
                report.protected += 1
                continue
            expire.append(r["id"])
        report.expired_rows = len(expire)
        expire_set = set(expire)
        if not dry_run and expire:
            with self.db.tx() as conn:
                conn.executemany(
                    "UPDATE artifacts SET state = 'expired' WHERE id = ?", [(i,) for i in expire]
                )
        referenced = {
            r["sha256"]
            for r in self.db.query("SELECT id, sha256 FROM artifacts WHERE state <> 'expired'")
            if r["id"] not in expire_set
        }
        for sub in self.root.iterdir():
            if sub.name == "tmp" or not sub.is_dir():
                continue
            for blob in sub.iterdir():
                if blob.name not in referenced:
                    report.removed_blobs += 1
                    report.freed_bytes += blob.stat().st_size
                    if not dry_run:
                        blob.unlink(missing_ok=True)
        for tmp in (self.root / "tmp").iterdir():
            if tmp.stat().st_mtime < time.time() - 3600:
                report.removed_temp += 1
                if not dry_run:
                    tmp.unlink(missing_ok=True)
        if not dry_run and expire:
            with self.db.tx() as conn:
                conn.executemany(
                    "DELETE FROM artifacts WHERE id = ? AND state = 'expired'", [(i,) for i in expire]
                )
        return report
