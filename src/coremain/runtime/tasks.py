"""TaskService: the only writer of task lifecycle state."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from typing import Any

from coremain.domain.models import Attempt, GateResult, Task
from coremain.domain.states import (
    ACTIVE,
    ALLOWED_AFTER_CANCEL,
    RESUMABLE,
    TERMINAL,
    TRANSITIONS,
    AttemptStatus,
    TaskStatus,
)
from coremain.errors import (
    ConflictError,
    GateFailedError,
    InvalidTransitionError,
    NotFoundError,
    OperationCancelled,
)
from coremain.events import EventLog
from coremain.runtime.leases import Fence, LeaseManager
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps

Gate = Callable[[sqlite3.Connection, Task], GateResult]


def task_resource(task_id: str) -> str:
    return f"task:{task_id}"


class TaskService:
    def __init__(self, db: Database, events: EventLog, leases: LeaseManager, clock: Clock):
        self.db = db
        self.events = events
        self.leases = leases
        self.clock = clock

    # ----------------------------------------------------------------- queries
    def _get(self, conn: sqlite3.Connection, task_id: str) -> Task:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"task {task_id} not found")
        return Task.from_row(row)

    def get(self, task_id: str) -> Task:
        return self._get(self.db.conn(), task_id)

    def resolve(self, fragment: str, *, project_id: str | None = None) -> Task:
        fragment = fragment.strip()
        row = self.db.one("SELECT * FROM tasks WHERE id = ?", (fragment,))
        if row is not None:
            return Task.from_row(row)
        body = fragment.split("_", 1)[1] if fragment.startswith("tsk_") else fragment
        sql = "SELECT * FROM tasks WHERE (id LIKE ? OR id LIKE ?)"
        params: list[Any] = [f"tsk_{body}%", f"%{body}"]
        if project_id:
            sql += " AND project_id = ?"
            params.append(project_id)
        rows = self.db.query(sql + " LIMIT 3", params)
        if not rows:
            raise NotFoundError(f"no task matches '{fragment}'")
        if len(rows) > 1:
            raise ConflictError(f"'{fragment}' is ambiguous: {', '.join(r['id'] for r in rows)}")
        return Task.from_row(rows[0])

    def list(
        self,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        statuses: Iterable[TaskStatus] | None = None,
        parent_task_id: str | None = None,
        roots_only: bool = False,
        limit: int = 50,
    ) -> list[Task]:
        sql = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []
        if project_id:
            sql += " AND project_id = ?"
            params.append(project_id)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if parent_task_id:
            sql += " AND parent_task_id = ?"
            params.append(parent_task_id)
        if roots_only:
            sql += " AND parent_task_id IS NULL"
        if statuses:
            values = [s.value for s in statuses]
            sql += f" AND status IN ({','.join('?' * len(values))})"
            params.extend(values)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [Task.from_row(r) for r in self.db.query(sql, params)]

    def attempts(self, task_id: str) -> list[Attempt]:
        return [
            Attempt.from_row(r)
            for r in self.db.query("SELECT * FROM attempts WHERE task_id = ? ORDER BY number", (task_id,))
        ]

    def attempt(self, attempt_id: str) -> Attempt:
        row = self.db.one("SELECT * FROM attempts WHERE id = ?", (attempt_id,))
        if row is None:
            raise NotFoundError(f"attempt {attempt_id} not found")
        return Attempt.from_row(row)

    def dependencies(self, task_id: str) -> list[Task]:
        rows = self.db.query(
            "SELECT t.* FROM task_deps d JOIN tasks t ON t.id = d.depends_on WHERE d.task_id = ?", (task_id,)
        )
        return [Task.from_row(r) for r in rows]

    def dependencies_met(self, task_id: str) -> bool:
        return all(dep.status == TaskStatus.COMPLETED for dep in self.dependencies(task_id))

    def runnable(self, *, project_id: str | None = None, limit: int = 20) -> list[Task]:
        sql = (
            "SELECT t.* FROM tasks t WHERE t.status = 'queued' AND t.cancel_requested_at IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM task_deps d JOIN tasks x ON x.id = d.depends_on "
            "WHERE d.task_id = t.id AND x.status <> 'completed')"
        )
        params: list[Any] = []
        if project_id:
            sql += " AND t.project_id = ?"
            params.append(project_id)
        sql += " ORDER BY t.priority DESC, t.created_at LIMIT ?"
        params.append(limit)
        return [Task.from_row(r) for r in self.db.query(sql, params)]

    # ----------------------------------------------------------------- creation
    def create(
        self,
        *,
        project_id: str,
        title: str,
        description: str,
        kind: str,
        session_id: str | None = None,
        parent_task_id: str | None = None,
        contract: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        mode: str | None = None,
        depends_on: Iterable[str] = (),
        priority: int = 0,
        idempotency_key: str | None = None,
        status: TaskStatus = TaskStatus.QUEUED,
    ) -> Task:
        now = self.clock.now()
        with self.db.tx() as conn:
            if idempotency_key:
                existing = conn.execute(
                    "SELECT * FROM tasks WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing is not None:
                    return Task.from_row(existing)
            depth = 0
            if parent_task_id:
                parent = self._get(conn, parent_task_id)
                depth = parent.depth + 1
            deps = list(depends_on)
            if deps and status == TaskStatus.QUEUED:
                status = TaskStatus.PENDING
            task_id = new_id("tsk", now=now)
            conn.execute(
                "INSERT INTO tasks(id, project_id, session_id, parent_task_id, kind, title, description, status, mode, "
                "contract_json, options_json, priority, depth, idempotency_key, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    project_id,
                    session_id,
                    parent_task_id,
                    kind,
                    title,
                    description,
                    status.value,
                    mode,
                    dumps(contract or {}),
                    dumps(options or {}),
                    priority,
                    depth,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            for dep in deps:
                conn.execute("INSERT INTO task_deps(task_id, depends_on) VALUES (?, ?)", (task_id, dep))
            self.events.emit(
                "task.created",
                project_id=project_id,
                session_id=session_id,
                task_id=task_id,
                data={
                    "title": title,
                    "kind": kind,
                    "status": status.value,
                    "parent": parent_task_id,
                    "depends_on": deps,
                },
            )
            return self._get(conn, task_id)

    def update_fields(self, task_id: str, *, fence: Fence | None = None, **fields: Any) -> Task:
        allowed = {
            "mode",
            "decision",
            "contract",
            "options",
            "workspace_id",
            "result_summary",
            "evidence_level",
            "title",
            "kind",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"cannot update task fields {sorted(unknown)}")
        sets, params = [], []
        for key, value in fields.items():
            column = f"{key}_json" if key in {"decision", "contract", "options"} else key
            sets.append(f"{column} = ?")
            params.append(dumps(value) if key in {"decision", "contract", "options"} else value)
        with self.db.tx() as conn:
            if fence:
                self.leases.verify(conn, fence)
            conn.execute(
                f"UPDATE tasks SET {', '.join(sets)}, version = version + 1, updated_at = ? WHERE id = ?",
                (*params, self.clock.now(), task_id),
            )
            return self._get(conn, task_id)

    # --------------------------------------------------------------- transitions
    def transition(
        self,
        task_id: str,
        to: TaskStatus,
        *,
        reason: str,
        actor: str = "runtime",
        fence: Fence | None = None,
        expect: Iterable[TaskStatus] | None = None,
        block_reason: str | None = None,
        result_summary: str | None = None,
        evidence_level: str | None = None,
        gate: Gate | None = None,
        data: dict[str, Any] | None = None,
    ) -> Task:
        with self.db.tx() as conn:
            task = self._get(conn, task_id)
            if fence is not None:
                self.leases.verify(conn, fence)
            src = task.status
            if src == to:
                return task
            if expect is not None and src not in set(expect):
                raise InvalidTransitionError(
                    f"task {task_id} is {src.value}, expected one of {[s.value for s in expect]}"
                )
            if to not in TRANSITIONS[src]:
                raise InvalidTransitionError(
                    f"illegal transition {src.value} → {to.value} for task {task_id}",
                    details={"from": src.value, "to": to.value},
                )
            if task.cancel_requested_at is not None and to not in ALLOWED_AFTER_CANCEL:
                raise OperationCancelled(
                    f"task {task_id} has a pending cancellation; refusing {src.value} → {to.value}"
                )
            gate_result: GateResult | None = None
            if to == TaskStatus.COMPLETED:
                if gate is None:
                    raise GateFailedError("completion requires an evidence gate evaluation")
                gate_result = gate(conn, task)
                if not gate_result.passed:
                    raise GateFailedError(
                        f"evidence gate failed: {gate_result.summary}", details=gate_result.to_dict()
                    )
                evidence_level = evidence_level or gate_result.level
            now = self.clock.now()
            cur = conn.execute(
                "UPDATE tasks SET status = ?, status_reason = ?, block_reason = ?, "
                "result_summary = COALESCE(?, result_summary), evidence_level = COALESCE(?, evidence_level), "
                "completed_at = CASE WHEN ? THEN ? ELSE completed_at END, "
                "pause_requested_at = CASE WHEN ? = 'paused' THEN NULL ELSE pause_requested_at END, "
                "version = version + 1, updated_at = ? WHERE id = ? AND version = ?",
                (
                    to.value,
                    reason,
                    block_reason if to == TaskStatus.BLOCKED else None,
                    result_summary,
                    evidence_level,
                    1 if to in TERMINAL else 0,
                    now,
                    to.value,
                    now,
                    task_id,
                    task.version,
                ),
            )
            if cur.rowcount != 1:
                raise ConflictError(f"task {task_id} was modified concurrently")
            payload: dict[str, Any] = {"from": src.value, "to": to.value, "reason": reason}
            if block_reason:
                payload["block_reason"] = block_reason
            if gate_result is not None:
                payload["gate"] = gate_result.to_dict()
            if data:
                payload.update(data)
            self.events.emit(
                "task.transition",
                project_id=task.project_id,
                session_id=task.session_id,
                task_id=task_id,
                actor=actor,
                level="warning"
                if to in {TaskStatus.FAILED, TaskStatus.UNKNOWN, TaskStatus.BLOCKED}
                else "info",
                data=payload,
            )
            return self._get(conn, task_id)

    def request_cancel(self, task_id: str, *, reason: str = "cancelled by user", actor: str = "user") -> Task:
        """Durably record cancellation. Idle tasks are cancelled immediately; active work is
        cancelled by its worker, which observes the flag and its in-process cancel token."""
        with self.db.tx() as conn:
            task = self._get(conn, task_id)
            if task.status in TERMINAL:
                return task
            now = self.clock.now()
            conn.execute(
                "UPDATE tasks SET cancel_requested_at = COALESCE(cancel_requested_at, ?), cancel_reason = ?, "
                "version = version + 1, updated_at = ? WHERE id = ?",
                (now, reason, now, task_id),
            )
            self.events.emit(
                "task.cancel_requested",
                project_id=task.project_id,
                session_id=task.session_id,
                task_id=task_id,
                actor=actor,
                data={"reason": reason},
            )
            holder = self.leases.holder(task_resource(task_id))
            live_worker = (
                holder is not None and holder["owner"] is not None and (holder["expires_at"] or 0) > now
            )
            waiting_inline = (
                task.status in {TaskStatus.AWAITING_APPROVAL, TaskStatus.NEEDS_INPUT} and live_worker
            )
            if task.status not in ACTIVE and not waiting_inline:
                self.transition(task_id, TaskStatus.CANCELLED, reason=reason, actor=actor)
                conn.execute(
                    "UPDATE approvals SET status = 'cancelled', decided_at = ? WHERE task_id = ? AND status = 'pending'",
                    (now, task_id),
                )
            for child in conn.execute("SELECT id FROM tasks WHERE parent_task_id = ?", (task_id,)).fetchall():
                self.request_cancel(child["id"], reason=f"parent cancelled: {reason}", actor=actor)
            return self._get(conn, task_id)

    def request_pause(self, task_id: str, *, actor: str = "user") -> Task:
        with self.db.tx() as conn:
            task = self._get(conn, task_id)
            if task.status in TERMINAL:
                return task
            if task.status == TaskStatus.QUEUED:
                return self.transition(task_id, TaskStatus.PAUSED, reason="paused by user", actor=actor)
            conn.execute(
                "UPDATE tasks SET pause_requested_at = ?, updated_at = ? WHERE id = ?",
                (self.clock.now(), self.clock.now(), task_id),
            )
            self.events.emit(
                "task.pause_requested",
                project_id=task.project_id,
                session_id=task.session_id,
                task_id=task_id,
                actor=actor,
            )
            return self._get(conn, task_id)

    def resume(self, task_id: str, *, actor: str = "user", note: str | None = None) -> Task:
        task = self.get(task_id)
        if task.cancel_requested_at is not None:
            raise InvalidTransitionError(
                f"task {task_id} was cancelled and cannot be resumed; create a new task instead"
            )
        if task.status not in RESUMABLE and task.status != TaskStatus.QUEUED:
            raise InvalidTransitionError(
                f"task {task_id} is {task.status.value}; only {sorted(s.value for s in RESUMABLE)} can be resumed"
            )
        with self.db.tx() as conn:
            conn.execute("UPDATE tasks SET pause_requested_at = NULL WHERE id = ?", (task_id,))
            return self.transition(task_id, TaskStatus.QUEUED, reason=note or "resumed", actor=actor)

    # ------------------------------------------------------------------ attempts
    def begin_attempt(
        self,
        task_id: str,
        fence: Fence,
        *,
        runtime_id: str,
        workflow: str | None = None,
        workspace_id: str | None = None,
        resumed_from: str | None = None,
        base_fingerprint: str | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> tuple[Task, Attempt]:
        now = self.clock.now()
        with self.db.tx() as conn:
            self.leases.verify(conn, fence)
            task = self.transition(
                task_id, TaskStatus.RUNNING, reason="attempt started", fence=fence, expect={TaskStatus.QUEUED}
            )
            number = int(
                conn.execute(
                    "SELECT COALESCE(MAX(number), 0) + 1 FROM attempts WHERE task_id = ?", (task_id,)
                ).fetchone()[0]
            )
            attempt_id = new_id("att", now=now)
            conn.execute(
                "INSERT INTO attempts(id, task_id, number, status, runtime_id, fence_token, workspace_id, workflow, "
                "checkpoint_json, base_fingerprint, resumed_from, started_at, heartbeat_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    task_id,
                    number,
                    AttemptStatus.RUNNING.value,
                    runtime_id,
                    fence.token,
                    workspace_id,
                    workflow,
                    dumps(checkpoint or {}),
                    base_fingerprint,
                    resumed_from,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE tasks SET attempt_count = attempt_count + 1, recovered_count = recovered_count + ? WHERE id = ?",
                (1 if resumed_from else 0, task_id),
            )
            self.events.emit(
                "attempt.started",
                project_id=task.project_id,
                session_id=task.session_id,
                task_id=task_id,
                attempt_id=attempt_id,
                data={
                    "number": number,
                    "fence": fence.token,
                    "resumed_from": resumed_from,
                    "workflow": workflow,
                },
            )
            return self._get(conn, task_id), Attempt.from_row(
                conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()
            )

    def end_attempt(
        self,
        attempt_id: str,
        status: AttemptStatus,
        *,
        fence: Fence | None,
        error_class: str | None = None,
        error_message: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> Attempt:
        with self.db.tx() as conn:
            if fence is not None:
                self.leases.verify(conn, fence)
            row = conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"attempt {attempt_id} not found")
            attempt = Attempt.from_row(row)
            if attempt.status not in {AttemptStatus.RUNNING}:
                return attempt
            conn.execute(
                "UPDATE attempts SET status = ?, error_class = ?, error_message = ?, usage_json = COALESCE(?, usage_json), "
                "ended_at = ? WHERE id = ?",
                (
                    status.value,
                    error_class,
                    error_message,
                    dumps(usage) if usage is not None else None,
                    self.clock.now(),
                    attempt_id,
                ),
            )
            task = self._get(conn, attempt.task_id)
            self.events.emit(
                "attempt.ended",
                project_id=task.project_id,
                session_id=task.session_id,
                task_id=task.id,
                attempt_id=attempt_id,
                level="info" if status == AttemptStatus.SUCCEEDED else "warning",
                data={"status": status.value, "error_class": error_class, "error": error_message},
            )
            return Attempt.from_row(
                conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()
            )

    def save_checkpoint(self, attempt_id: str, fence: Fence, checkpoint: dict[str, Any]) -> None:
        with self.db.tx() as conn:
            self.leases.verify(conn, fence)
            conn.execute(
                "UPDATE attempts SET checkpoint_json = ?, heartbeat_at = ? WHERE id = ?",
                (dumps(checkpoint), self.clock.now(), attempt_id),
            )

    def record_usage(self, attempt_id: str, usage: dict[str, Any]) -> None:
        self.db.execute("UPDATE attempts SET usage_json = ? WHERE id = ?", (dumps(usage), attempt_id))

    def heartbeat(self, attempt_id: str, fence: Fence, ttl_s: float) -> bool:
        if not self.leases.renew(fence, ttl_s):
            return False
        self.db.execute("UPDATE attempts SET heartbeat_at = ? WHERE id = ?", (self.clock.now(), attempt_id))
        return True

    def release_dependents(self, task_id: str) -> list[str]:
        """Move pending tasks whose dependencies are now all completed into the queue."""
        released: list[str] = []
        rows = self.db.query("SELECT task_id FROM task_deps WHERE depends_on = ?", (task_id,))
        for r in rows:
            dep_task = self.get(r["task_id"])
            if dep_task.status == TaskStatus.PENDING and self.dependencies_met(dep_task.id):
                self.transition(dep_task.id, TaskStatus.QUEUED, reason="dependencies completed")
                released.append(dep_task.id)
        return released
