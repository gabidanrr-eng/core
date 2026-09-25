"""Layered memory.

Scopes: ``task`` (current execution), ``session`` (active work), ``project`` (stable facts and
decisions), ``operational`` (recurring tool/environment behaviour) and ``global`` (only when
explicitly enabled). Kinds separate *facts* from *hypotheses*, *suggestions*, *preferences*,
*observations*, *procedures* and *decisions*.

Authority rules: only the user or verified evidence can create facts. Anything a model
proposes is stored as a hypothesis/suggestion/observation until promoted by the user or by
evidence. Items may be anchored to file hashes; when an anchored file changes the item is
marked stale and down-weighted in retrieval instead of silently remaining "true".
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coremain.config.schema import MemoryConfig
from coremain.errors import ConflictError, NotFoundError, PolicyDeniedError
from coremain.events import EventLog
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, loads, sha256_hex
from coremain.util.text import fts_query, query_terms

SCOPES = ("task", "session", "project", "operational", "global")
KINDS = ("fact", "hypothesis", "suggestion", "preference", "decision", "observation", "procedure")
AUTHORITATIVE = frozenset({"user", "evidence"})
DEFAULT_CONFIDENCE = {"user": 0.95, "evidence": 0.9, "runtime": 0.8, "tool": 0.7, "import": 0.6, "external": 0.5, "model": 0.4}
KIND_WEIGHT = {"fact": 1.0, "decision": 1.0, "preference": 0.95, "procedure": 0.9, "observation": 0.7, "hypothesis": 0.55, "suggestion": 0.5}


@dataclass
class MemoryItem:
    id: str
    project_id: str | None
    session_id: str | None
    task_id: str | None
    scope: str
    kind: str
    content: str
    tags: str
    confidence: float
    source_type: str
    source_ref: str | None
    evidence_ids: list[str]
    anchors: list[dict[str, str]]
    status: str
    status_reason: str | None
    created_at: float
    updated_at: float
    use_count: int = 0
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, r: Any) -> MemoryItem:
        return cls(r["id"], r["project_id"], r["session_id"], r["task_id"], r["scope"], r["kind"], r["content"], r["tags"],
                   r["confidence"], r["source_type"], r["source_ref"], loads(r["evidence_ids_json"], []), loads(r["anchors_json"], []),
                   r["status"], r["status_reason"], r["created_at"], r["updated_at"], r["use_count"])

    def label(self) -> str:
        basis = {"user": "stated by user", "evidence": "verified by evidence", "runtime": "observed by runtime", "tool": "tool output",
                 "model": "unverified model proposal", "external": "external source", "import": "imported"}.get(self.source_type, self.source_type)
        date = time.strftime("%Y-%m-%d", time.gmtime(self.updated_at))
        stale = ", STALE: " + (self.status_reason or "anchors changed") if self.status == "stale" else ""
        return f"[{self.kind} · {basis} · confidence {self.confidence:.2f} · {date}{stale}]"

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("notes", None)
        return d


class MemoryStore:
    def __init__(self, db: Database, events: EventLog, clock: Clock, config: MemoryConfig):
        self.db = db
        self.events = events
        self.clock = clock
        self.config = config

    def add(
        self,
        *,
        content: str,
        kind: str,
        scope: str,
        source_type: str,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        tags: Iterable[str] = (),
        confidence: float | None = None,
        source_ref: str | None = None,
        evidence_ids: Iterable[str] = (),
        anchors: Iterable[dict[str, str]] = (),
        valid_until: float | None = None,
    ) -> MemoryItem:
        if scope not in SCOPES:
            raise ValueError(f"unknown memory scope '{scope}'")
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind '{kind}'")
        content = content.strip()
        if not content:
            raise ValueError("memory content is empty")
        if scope == "global" and not self.config.global_enabled:
            raise PolicyDeniedError("global memory is disabled (memory.global_enabled = false)")
        if scope in {"project", "operational", "session", "task"} and not project_id:
            raise ValueError(f"{scope} memory requires a project")
        notes: list[str] = []
        evidence = list(evidence_ids)
        if kind == "fact" and source_type not in AUTHORITATIVE and not (source_type == "runtime" and evidence):
            kind = "hypothesis"
            notes.append("downgraded to hypothesis: facts require user confirmation or evidence")
        if source_type == "model" and kind in {"preference", "decision", "procedure"}:
            kind = "suggestion"
            notes.append("model-originated content stored as a suggestion")
        confidence = DEFAULT_CONFIDENCE.get(source_type, 0.5) if confidence is None else max(0.0, min(1.0, confidence))
        if source_type == "model":
            confidence = min(confidence, 0.6)
        now = self.clock.now()
        chash = sha256_hex(f"{scope}|{kind}|{' '.join(content.lower().split())}")
        tag_text = " ".join(sorted({t.strip().lower() for t in tags if t.strip()}))
        with self.db.tx() as conn:
            existing = conn.execute(
                "SELECT * FROM memory WHERE content_hash = ? AND COALESCE(project_id,'') = COALESCE(?, '') AND status = 'active'",
                (chash, project_id),
            ).fetchone()
            if existing is not None:
                merged = sorted(set(loads(existing["evidence_ids_json"], [])) | set(evidence))
                conn.execute("UPDATE memory SET evidence_ids_json = ?, confidence = MAX(confidence, ?), updated_at = ? WHERE id = ?",
                             (dumps(merged), confidence, now, existing["id"]))
                item = self.get(existing["id"])
                item.notes = ["deduplicated with an existing entry", *notes]
                return item
            mem_id = new_id("mem", now=now)
            conn.execute(
                "INSERT INTO memory(id, project_id, session_id, task_id, scope, kind, content, tags, confidence, source_type, source_ref, "
                "evidence_ids_json, anchors_json, content_hash, valid_until, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mem_id, project_id, session_id, task_id, scope, kind, content, tag_text, confidence, source_type, source_ref,
                 dumps(evidence), dumps(list(anchors)), chash, valid_until, now, now),
            )
            self.events.emit("memory.recorded", project_id=project_id, session_id=session_id, task_id=task_id,
                             data={"memory_id": mem_id, "scope": scope, "kind": kind, "source": source_type, "preview": content[:160]})
        item = self.get(mem_id)
        item.notes = notes
        return item

    def get(self, memory_id: str) -> MemoryItem:
        row = self.db.one("SELECT * FROM memory WHERE id = ?", (memory_id,))
        if row is None:
            raise NotFoundError(f"memory {memory_id} not found")
        return MemoryItem.from_row(row)

    def resolve(self, fragment: str) -> MemoryItem:
        rows = self.db.query("SELECT * FROM memory WHERE id = ? OR id LIKE ? LIMIT 3", (fragment, f"%{fragment}"))
        if not rows:
            raise NotFoundError(f"no memory matches '{fragment}'")
        if len(rows) > 1:
            raise ConflictError(f"'{fragment}' is ambiguous")
        return MemoryItem.from_row(rows[0])

    def list(self, project_id: str | None, *, scope: str | None = None, kind: str | None = None,
             statuses: tuple[str, ...] = ("active", "stale"), limit: int = 100) -> list[MemoryItem]:
        sql = f"SELECT * FROM memory WHERE status IN ({','.join('?' * len(statuses))})"
        params: list[Any] = list(statuses)
        if project_id is not None:
            sql += " AND (project_id = ? OR scope = 'global')"
            params.append(project_id)
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        return [MemoryItem.from_row(r) for r in self.db.query(sql + " ORDER BY updated_at DESC LIMIT ?", (*params, limit))]

    def search(self, project_id: str | None, query: str, *, scopes: tuple[str, ...] = ("project", "operational", "session"),
               session_id: str | None = None, limit: int = 12) -> list[MemoryItem]:
        if not self.config.enabled:
            return []
        allowed = [s for s in scopes if s != "global" or self.config.global_enabled]
        if self.config.global_enabled and "global" not in allowed:
            allowed.append("global")
        match = fts_query(query_terms(query))
        params: list[Any] = []
        scope_sql = f"m.scope IN ({','.join('?' * len(allowed))})"
        params.extend(allowed)
        project_sql = "(m.project_id = ? OR m.scope = 'global')"
        params.append(project_id)
        session_sql = ""
        if session_id:
            session_sql = " AND (m.scope <> 'session' OR m.session_id = ?)"
            params.append(session_id)
        now = self.clock.now()
        if match:
            rows = self.db.query(
                f"SELECT m.*, bm25(memory_fts) AS rank FROM memory_fts JOIN memory m ON m.seq = memory_fts.rowid "
                f"WHERE memory_fts MATCH ? AND {scope_sql} AND {project_sql}{session_sql} AND m.status IN ('active','stale') "
                "AND (m.valid_until IS NULL OR m.valid_until > ?) ORDER BY rank LIMIT ?",
                [match, *params, now, limit * 4],
            )
        else:
            rows = []
        pinned = self.db.query(
            f"SELECT m.*, 0.0 AS rank FROM memory m WHERE {scope_sql} AND {project_sql}{session_sql} AND m.status = 'active' "
            "AND m.kind IN ('preference','decision') AND m.source_type = 'user' ORDER BY m.updated_at DESC LIMIT 6",
            params,
        )
        seen: set[str] = set()
        items: list[MemoryItem] = []
        for r in [*rows, *pinned]:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            item = MemoryItem.from_row(r)
            relevance = 1.0 / (1.0 + max(0.0, float(r["rank"]) + 10.0) / 10.0) if match and r["rank"] else 0.5
            age_days = max(0.0, (now - item.updated_at) / 86400)
            recency = 1.0 / (1.0 + age_days / 90)
            item.score = relevance * KIND_WEIGHT.get(item.kind, 0.5) * (0.4 + 0.6 * item.confidence) * (0.6 + 0.4 * recency)
            if item.status == "stale":
                item.score *= 0.5
            items.append(item)
        items.sort(key=lambda i: i.score, reverse=True)
        return items[:limit]

    def set_status(self, memory_id: str, status: str, *, reason: str, actor: str = "user") -> MemoryItem:
        item = self.get(memory_id)
        self.db.execute("UPDATE memory SET status = ?, status_reason = ?, updated_at = ? WHERE id = ?",
                        (status, reason, self.clock.now(), memory_id))
        self.events.emit(f"memory.{status}", project_id=item.project_id, actor=actor, data={"memory_id": memory_id, "reason": reason})
        return self.get(memory_id)

    def invalidate(self, memory_id: str, *, reason: str, actor: str = "user") -> MemoryItem:
        return self.set_status(memory_id, "invalidated", reason=reason, actor=actor)

    def promote(self, memory_id: str, *, evidence_ids: Iterable[str] = (), by_user: bool = False) -> MemoryItem:
        item = self.get(memory_id)
        evidence = sorted(set(item.evidence_ids) | set(evidence_ids))
        if not by_user and not evidence:
            raise PolicyDeniedError("promotion to fact requires evidence or explicit user confirmation")
        source = "user" if by_user else "evidence"
        self.db.execute("UPDATE memory SET kind = 'fact', source_type = ?, confidence = MAX(confidence, ?), evidence_ids_json = ?, "
                        "status = 'active', status_reason = NULL, updated_at = ? WHERE id = ?",
                        (source, DEFAULT_CONFIDENCE[source], dumps(evidence), self.clock.now(), memory_id))
        self.events.emit("memory.promoted", project_id=item.project_id, data={"memory_id": memory_id, "source": source})
        return self.get(memory_id)

    def delete(self, memory_id: str) -> None:
        item = self.get(memory_id)
        self.db.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
        self.events.emit("memory.deleted", project_id=item.project_id, actor="user", data={"memory_id": memory_id})

    def mark_used(self, ids: Iterable[str]) -> None:
        now = self.clock.now()
        with self.db.tx() as conn:
            conn.executemany("UPDATE memory SET use_count = use_count + 1, last_used_at = ? WHERE id = ?", [(now, i) for i in ids])

    def check_anchors(self, project_id: str, root: Path) -> list[str]:
        stale: list[str] = []
        for item in self.list(project_id, statuses=("active",), limit=5000):
            for anchor in item.anchors:
                path = root / anchor.get("path", "")
                expected = anchor.get("sha256")
                try:
                    current = sha256_hex(path.read_bytes()) if path.is_file() else None
                except OSError:
                    current = None
                if expected and current != expected:
                    self.set_status(item.id, "stale", reason=f"anchored file changed: {anchor.get('path')}", actor="runtime")
                    stale.append(item.id)
                    break
        return stale

    def version(self, project_id: str | None) -> str:
        row = self.db.one("SELECT COUNT(*) AS n, COALESCE(MAX(updated_at), 0) AS t FROM memory WHERE project_id = ? OR scope = 'global'",
                          (project_id,))
        return f"{row['n']}:{row['t']}" if row else "0:0"
