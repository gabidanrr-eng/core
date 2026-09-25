"""Human approvals as a designed workflow state.

An approval is a durable record. Interactive clients (TUI, CLI prompt) register a handler
and answer in-process; any other process can answer with ``core approvals approve``. Waiters
observe decisions either through an in-process future or by polling the database, so an
approval survives terminal closure and crashes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from coremain.errors import ConflictError, NotFoundError
from coremain.events import EventLog
from coremain.runtime.cancel import CancelToken
from coremain.security.policy import PolicyEngine
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, loads

Scope = Literal["once", "task", "session", "project"]


@dataclass
class Approval:
    id: str
    project_id: str | None
    session_id: str | None
    task_id: str | None
    attempt_id: str | None
    tool_call_id: str | None
    capability: str
    summary: str
    request: dict[str, Any]
    status: str
    scope: str | None
    decided_by: str | None
    reason: str | None
    created_at: float
    decided_at: float | None

    @classmethod
    def from_row(cls, r: Any) -> Approval:
        return cls(r["id"], r["project_id"], r["session_id"], r["task_id"], r["attempt_id"], r["tool_call_id"], r["capability"],
                   r["summary"], loads(r["request_json"], {}), r["status"], r["scope"], r["decided_by"], r["reason"],
                   r["created_at"], r["decided_at"])

    @property
    def approved(self) -> bool:
        return self.status == "approved"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ApprovalService:
    def __init__(self, db: Database, events: EventLog, policy: PolicyEngine, clock: Clock):
        self.db = db
        self.events = events
        self.policy = policy
        self.clock = clock
        self._futures: dict[str, asyncio.Future[Approval]] = {}

    def request(self, *, capability: str, summary: str, request: dict[str, Any], project_id: str | None,
                session_id: str | None, task_id: str | None, attempt_id: str | None, tool_call_id: str | None) -> Approval:
        now = self.clock.now()
        approval_id = new_id("apr", now=now)
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO approvals(id, project_id, session_id, task_id, attempt_id, tool_call_id, capability, summary, request_json, "
                "status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (approval_id, project_id, session_id, task_id, attempt_id, tool_call_id, capability, summary,
                 dumps(request), "pending", now),
            )
            self.events.emit("approval.requested", project_id=project_id, session_id=session_id, task_id=task_id,
                             attempt_id=attempt_id, level="warning",
                             data={"approval_id": approval_id, "capability": capability, "summary": summary, "request": request})
        return self.get(approval_id)

    def get(self, approval_id: str) -> Approval:
        row = self.db.one("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        if row is None:
            raise NotFoundError(f"approval {approval_id} not found")
        return Approval.from_row(row)

    def resolve(self, fragment: str) -> Approval:
        row = self.db.one("SELECT * FROM approvals WHERE id = ? OR id LIKE ?", (fragment, f"%{fragment}"))
        if row is None:
            raise NotFoundError(f"no approval matches '{fragment}'")
        return Approval.from_row(row)

    def pending(self, *, project_id: str | None = None, task_id: str | None = None) -> list[Approval]:
        sql = "SELECT * FROM approvals WHERE status = 'pending'"
        params: list[Any] = []
        if project_id:
            sql += " AND project_id = ?"
            params.append(project_id)
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        return [Approval.from_row(r) for r in self.db.query(sql + " ORDER BY created_at", params)]

    def decide(self, approval_id: str, approved: bool, *, scope: Scope = "once", actor: str = "user",
               reason: str | None = None, pattern: str | None = None) -> Approval:
        approval = self.get(approval_id)
        if approval.status != "pending":
            if (approval.status == "approved") == approved:
                return approval
            raise ConflictError(f"approval {approval_id} was already {approval.status}")
        now = self.clock.now()
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE approvals SET status = ?, scope = ?, decided_by = ?, reason = ?, decided_at = ? WHERE id = ? AND status = 'pending'",
                ("approved" if approved else "rejected", scope, actor, reason, now, approval_id),
            )
            if scope != "once":
                target = pattern or str(approval.request.get("target", "*"))
                self.policy.grant(
                    approval.capability, pattern=target, decision="allow" if approved else "deny", granted_by="approval",
                    project_id=approval.project_id,
                    session_id=approval.session_id if scope in ("session", "task") else None,
                    task_id=approval.task_id if scope == "task" else None,
                    reason=f"approval {approval_id} ({scope})",
                )
            self.events.emit("approval.decided", project_id=approval.project_id, session_id=approval.session_id,
                             task_id=approval.task_id, attempt_id=approval.attempt_id, actor=actor,
                             data={"approval_id": approval_id, "approved": approved, "scope": scope, "reason": reason})
        decided = self.get(approval_id)
        fut = self._futures.pop(approval_id, None)
        if fut is not None and not fut.done():
            fut.get_loop().call_soon_threadsafe(fut.set_result, decided)
        return decided

    async def wait(self, approval_id: str, cancel: CancelToken, *, poll_s: float = 0.5) -> Approval:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Approval] = loop.create_future()
        self._futures[approval_id] = fut
        try:
            while True:
                current = self.get(approval_id)
                if current.status != "pending":
                    return current
                cancel.raise_if_cancelled()
                try:
                    return await asyncio.wait_for(asyncio.shield(fut), timeout=poll_s)
                except TimeoutError:
                    continue
        finally:
            self._futures.pop(approval_id, None)

    def cancel_for_task(self, task_id: str) -> None:
        self.db.execute("UPDATE approvals SET status = 'cancelled', decided_at = ? WHERE task_id = ? AND status = 'pending'",
                        (self.clock.now(), task_id))
