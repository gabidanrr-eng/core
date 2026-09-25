"""Architecture decision records with evidence links and supersession."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from coremain.errors import ConflictError, NotFoundError
from coremain.events import EventLog
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, loads
from coremain.util.text import query_terms


@dataclass
class Decision:
    id: str
    project_id: str
    title: str
    status: str
    context: str
    decision: str
    alternatives: list[str]
    consequences: str | None
    evidence_ids: list[str]
    source: str
    supersedes_id: str | None
    created_at: float

    @classmethod
    def from_row(cls, r: Any) -> Decision:
        return cls(r["id"], r["project_id"], r["title"], r["status"], r["context"], r["decision"], loads(r["alternatives_json"], []),
                   r["consequences"], loads(r["evidence_ids_json"], []), r["source"], r["supersedes_id"], r["created_at"])

    def render(self) -> str:
        alts = "; ".join(self.alternatives) if self.alternatives else "none recorded"
        return (f"ADR {self.id} [{self.status}] {self.title}\nContext: {self.context}\nDecision: {self.decision}\n"
                f"Rejected alternatives: {alts}" + (f"\nConsequences: {self.consequences}" if self.consequences else ""))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class DecisionStore:
    def __init__(self, db: Database, events: EventLog, clock: Clock):
        self.db = db
        self.events = events
        self.clock = clock

    def add(self, project_id: str, *, title: str, context: str, decision: str, alternatives: Iterable[str] = (),
            consequences: str | None = None, status: str = "accepted", source: str = "user", evidence_ids: Iterable[str] = (),
            supersedes_id: str | None = None, task_id: str | None = None, session_id: str | None = None) -> Decision:
        now = self.clock.now()
        adr_id = new_id("adr", now=now)
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO decisions(id, project_id, session_id, task_id, title, status, context, decision, alternatives_json, consequences, "
                "evidence_ids_json, source, supersedes_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (adr_id, project_id, session_id, task_id, title, status, context, decision, dumps(list(alternatives)), consequences,
                 dumps(list(evidence_ids)), source, supersedes_id, now, now),
            )
            if supersedes_id:
                conn.execute("UPDATE decisions SET status = 'superseded', updated_at = ? WHERE id = ?", (now, supersedes_id))
            self.events.emit("decision.recorded", project_id=project_id, data={"decision_id": adr_id, "title": title, "status": status})
        return self.get(adr_id)

    def get(self, decision_id: str) -> Decision:
        row = self.db.one("SELECT * FROM decisions WHERE id = ?", (decision_id,))
        if row is None:
            raise NotFoundError(f"decision {decision_id} not found")
        return Decision.from_row(row)

    def resolve(self, fragment: str) -> Decision:
        rows = self.db.query("SELECT * FROM decisions WHERE id = ? OR id LIKE ? LIMIT 3", (fragment, f"%{fragment}"))
        if not rows:
            raise NotFoundError(f"no decision matches '{fragment}'")
        if len(rows) > 1:
            raise ConflictError(f"'{fragment}' is ambiguous")
        return Decision.from_row(rows[0])

    def list(self, project_id: str, *, status: str | None = None) -> list[Decision]:
        sql = "SELECT * FROM decisions WHERE project_id = ?" + (" AND status = ?" if status else "") + " ORDER BY created_at DESC"
        return [Decision.from_row(r) for r in self.db.query(sql, (project_id, status) if status else (project_id,))]

    def set_status(self, decision_id: str, status: str) -> Decision:
        if status not in {"proposed", "accepted", "rejected", "superseded"}:
            raise ValueError(f"invalid decision status '{status}'")
        self.db.execute("UPDATE decisions SET status = ?, updated_at = ? WHERE id = ?", (status, self.clock.now(), decision_id))
        return self.get(decision_id)

    def relevant(self, project_id: str, text: str, *, limit: int = 5) -> list[Decision]:
        terms = query_terms(text, max_terms=12)
        decisions = self.list(project_id, status="accepted")
        if not terms:
            return decisions[:limit]
        scored = []
        for d in decisions:
            blob = f"{d.title} {d.context} {d.decision}".lower()
            score = sum(1 for t in terms if t in blob)
            if score:
                scored.append((score, d))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [d for _, d in scored[:limit]]
