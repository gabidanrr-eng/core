"""Project identity and engineering sessions."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from coremain.domain.models import Message, Project, Session
from coremain.errors import ConflictError, NotFoundError
from coremain.events import EventLog
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps
from coremain.util.text import fts_query, one_line, query_terms


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


class ProjectService:
    def __init__(self, db: Database, events: EventLog, clock: Clock):
        self.db = db
        self.events = events
        self.clock = clock

    def ensure(self, root: Path) -> Project:
        root = root.resolve()
        row = self.db.one("SELECT * FROM projects WHERE root_path = ?", (str(root),))
        if row is not None:
            return Project.from_row(row)
        is_git = _git(root, "rev-parse", "--is-inside-work-tree") == "true"
        remote = _git(root, "config", "--get", "remote.origin.url") if is_git else None
        now = self.clock.now()
        project_id = new_id("prj", now=now)
        with self.db.tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO projects(id, root_path, name, vcs, git_remote, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (project_id, str(root), root.name, "git" if is_git else "none", remote, now, now),
            )
            row = conn.execute("SELECT * FROM projects WHERE root_path = ?", (str(root),)).fetchone()
            self.events.emit(
                "project.registered", project_id=row["id"], data={"root": str(root), "vcs": row["vcs"]}
            )
        return Project.from_row(row)

    def get(self, project_id: str) -> Project:
        row = self.db.one("SELECT * FROM projects WHERE id = ?", (project_id,))
        if row is None:
            raise NotFoundError(f"project {project_id} not found")
        return Project.from_row(row)

    def list(self) -> list[Project]:
        return [Project.from_row(r) for r in self.db.query("SELECT * FROM projects ORDER BY updated_at DESC")]

    def set_trusted(self, project_id: str, config_hash: str | None) -> None:
        with self.db.tx():
            self.db.execute(
                "UPDATE projects SET trusted_config_hash = ?, trusted_at = ?, updated_at = ? WHERE id = ?",
                (config_hash or "none", self.clock.now(), self.clock.now(), project_id),
            )
            self.events.emit(
                "project.trusted", project_id=project_id, actor="user", data={"config_hash": config_hash}
            )

    def revoke_trust(self, project_id: str) -> None:
        with self.db.tx():
            self.db.execute(
                "UPDATE projects SET trusted_config_hash = NULL, trusted_at = NULL WHERE id = ?",
                (project_id,),
            )
            self.events.emit("project.untrusted", project_id=project_id, actor="user")

    @staticmethod
    def is_trusted(project: Project, current_config_hash: str | None) -> bool:
        if project.trusted_config_hash is None:
            return False
        return project.trusted_config_hash == (current_config_hash or "none")

    def update_profile(self, project_id: str, profile: dict[str, Any], profile_hash: str) -> None:
        now = self.clock.now()
        self.db.execute(
            "UPDATE projects SET profile_json = ?, profile_hash = ?, profile_updated_at = ?, updated_at = ? WHERE id = ?",
            (dumps(profile), profile_hash, now, now, project_id),
        )

    def touch(self, project_id: str) -> None:
        self.db.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (self.clock.now(), project_id))


class SessionService:
    def __init__(self, db: Database, events: EventLog, clock: Clock):
        self.db = db
        self.events = events
        self.clock = clock

    def create(
        self,
        project_id: str,
        title: str = "New session",
        *,
        parent_session_id: str | None = None,
        fork_checkpoint_id: str | None = None,
        intent: str | None = None,
    ) -> Session:
        now = self.clock.now()
        session_id = new_id("ses", now=now)
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO sessions(id, project_id, title, parent_session_id, fork_checkpoint_id, intent, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (session_id, project_id, title, parent_session_id, fork_checkpoint_id, intent, now, now),
            )
            self.events.emit(
                "session.created",
                project_id=project_id,
                session_id=session_id,
                data={"title": title, "parent": parent_session_id},
            )
        return self.get(session_id)

    def get(self, session_id: str) -> Session:
        row = self.db.one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            raise NotFoundError(f"session {session_id} not found")
        return Session.from_row(row)

    def resolve(self, fragment: str, *, project_id: str | None = None) -> Session:
        row = self.db.one("SELECT * FROM sessions WHERE id = ?", (fragment,))
        if row is not None:
            return Session.from_row(row)
        body = fragment.split("_", 1)[1] if fragment.startswith("ses_") else fragment
        sql = "SELECT * FROM sessions WHERE (id LIKE ? OR id LIKE ?)"
        params: list[Any] = [f"ses_{body}%", f"%{body}"]
        if project_id:
            sql += " AND project_id = ?"
            params.append(project_id)
        rows = self.db.query(sql + " LIMIT 3", params)
        if not rows:
            raise NotFoundError(f"no session matches '{fragment}'")
        if len(rows) > 1:
            raise ConflictError(f"'{fragment}' is ambiguous")
        return Session.from_row(rows[0])

    def latest(self, project_id: str) -> Session | None:
        row = self.db.one(
            "SELECT * FROM sessions WHERE project_id = ? AND status = 'active' ORDER BY updated_at DESC LIMIT 1",
            (project_id,),
        )
        return Session.from_row(row) if row else None

    def list(self, project_id: str, *, include_archived: bool = False, limit: int = 100) -> list[Session]:
        sql = "SELECT * FROM sessions WHERE project_id = ?" + (
            "" if include_archived else " AND status = 'active'"
        )
        return [
            Session.from_row(r)
            for r in self.db.query(sql + " ORDER BY updated_at DESC LIMIT ?", (project_id, limit))
        ]

    def rename(self, session_id: str, title: str) -> None:
        self.db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, self.clock.now(), session_id),
        )

    def set_summary(self, session_id: str, summary: str) -> None:
        self.db.execute(
            "UPDATE sessions SET summary = ?, updated_at = ? WHERE id = ?",
            (summary, self.clock.now(), session_id),
        )

    def archive(self, session_id: str) -> None:
        session = self.get(session_id)
        with self.db.tx():
            self.db.execute(
                "UPDATE sessions SET status = 'archived', updated_at = ? WHERE id = ?",
                (self.clock.now(), session_id),
            )
            self.events.emit(
                "session.archived", project_id=session.project_id, session_id=session_id, actor="user"
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        task_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        now = self.clock.now()
        message_id = new_id("msg", now=now)
        session = self.get(session_id)
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO messages(id, session_id, task_id, role, content, meta_json, created_at) VALUES (?,?,?,?,?,?,?)",
                (message_id, session_id, task_id, role, content, dumps(meta or {}), now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            if role == "user" and session.title == "New session":
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?", (one_line(content, 60), session_id)
                )
            self.events.emit(
                "message.created",
                project_id=session.project_id,
                session_id=session_id,
                task_id=task_id,
                data={"message_id": message_id, "role": role, "preview": one_line(content, 200)},
            )
        row = self.db.one("SELECT * FROM messages WHERE id = ?", (message_id,))
        assert row is not None
        return Message.from_row(row)

    def messages(self, session_id: str, *, limit: int = 500) -> list[Message]:
        rows = self.db.query(
            "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?) ORDER BY seq",
            (session_id, limit),
        )
        return [Message.from_row(r) for r in rows]

    def search(self, project_id: str, query: str, *, limit: int = 20) -> list[tuple[Message, str]]:
        match = fts_query(query_terms(query))
        if not match:
            return []
        rows = self.db.query(
            "SELECT m.*, snippet(messages_fts, 0, '[', ']', '…', 12) AS snip FROM messages_fts "
            "JOIN messages m ON m.seq = messages_fts.rowid JOIN sessions s ON s.id = m.session_id "
            "WHERE messages_fts MATCH ? AND s.project_id = ? ORDER BY bm25(messages_fts) LIMIT ?",
            (match, project_id, limit),
        )
        return [(Message.from_row(r), r["snip"]) for r in rows]
