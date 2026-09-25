"""Crash recovery and reconciliation of durable state with reality.

Runs automatically when a runtime opens (non-destructive parts) and on demand via
``core recover``. Work is never silently resumed or discarded: interrupted attempts become
``interrupted`` (resumable) or ``unknown`` (needs inspection) and orphaned processes are
reported, and terminated only when explicitly requested.
"""

from __future__ import annotations

import os
import signal
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from coremain.domain.states import ACTIVE, AttemptStatus, TaskStatus
from coremain.errors import CoreError
from coremain.exec.process import kill_group, process_alive

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime

RUNTIME_STALE_S = 90.0


@dataclass
class RecoveryReport:
    stale_runtimes: list[str] = field(default_factory=list)
    interrupted_tasks: list[dict[str, Any]] = field(default_factory=list)
    unknown_tasks: list[dict[str, Any]] = field(default_factory=list)
    orphan_processes: list[dict[str, Any]] = field(default_factory=list)
    killed_processes: list[int] = field(default_factory=list)
    interrupted_tool_calls: int = 0
    cancelled_model_calls: int = 0
    orphaned_workspaces: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(
            self.stale_runtimes
            or self.interrupted_tasks
            or self.unknown_tasks
            or self.killed_processes
            or self.interrupted_tool_calls
            or self.orphaned_workspaces
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def recover(rt: CoreRuntime, *, kill_orphans: bool = False, dry_run: bool = False) -> RecoveryReport:
    report = RecoveryReport()
    db = rt.db
    now = rt.clock.now()
    host = socket.gethostname()
    for r in db.query("SELECT * FROM runtimes WHERE status = 'running' AND id <> ?", (rt.runtime_id,)):
        same_host_dead = r["host"] == host and not process_alive(int(r["pid"]))
        silent_too_long = (now - float(r["heartbeat_at"])) > RUNTIME_STALE_S * 4
        if same_host_dead or silent_too_long:
            report.stale_runtimes.append(r["id"])
            if not dry_run:
                db.execute("UPDATE runtimes SET status = 'dead', stopped_at = ? WHERE id = ?", (now, r["id"]))
    dead_runtimes = {r["id"] for r in db.query("SELECT id FROM runtimes WHERE status <> 'running'")} | set(
        report.stale_runtimes
    )
    # A lease held by a runtime proven dead is reclaimable now; fencing tokens still reject any
    # late write from the old holder, so this cannot corrupt state even if the proof were wrong.
    candidates = {(r["resource"], r["token"]): r for r in rt.leases.expired()}
    candidates.update({(r["resource"], r["token"]): r for r in rt.leases.held_by_runtimes(dead_runtimes)})
    for lease in candidates.values():
        resource = lease["resource"]
        if not resource.startswith("task:"):
            continue
        task_id = resource.split(":", 1)[1]
        try:
            task = rt.tasks.get(task_id)
        except CoreError:
            continue
        attempt = db.one(
            "SELECT * FROM attempts WHERE task_id = ? AND fence_token = ? ORDER BY number DESC LIMIT 1",
            (task_id, lease["token"]),
        )
        if not dry_run:
            db.execute(
                "UPDATE leases SET owner = NULL, released_at = ? WHERE resource = ? AND token = ?",
                (now, resource, lease["token"]),
            )
        if attempt is None or attempt["status"] != AttemptStatus.RUNNING.value:
            continue
        checkpoint_node = None
        try:
            import json

            checkpoint_node = json.loads(attempt["checkpoint_json"] or "{}").get("node")
        except ValueError:
            checkpoint_node = None
        ambiguous = checkpoint_node == "finalize"
        entry = {
            "task": task_id,
            "attempt": attempt["id"],
            "node": checkpoint_node,
            "status": task.status.value,
        }
        if dry_run:
            (report.unknown_tasks if ambiguous else report.interrupted_tasks).append(entry)
            continue
        db.execute(
            "UPDATE attempts SET status = ?, ended_at = ?, error_class = 'interrupted', error_message = ? WHERE id = ?",
            (
                AttemptStatus.INTERRUPTED.value,
                now,
                "owner runtime stopped without finishing (lease expired)",
                attempt["id"],
            ),
        )
        report.interrupted_tool_calls += db.execute(
            "UPDATE tool_calls SET status = 'interrupted', ended_at = ? WHERE attempt_id = ? AND status = 'running'",
            (now, attempt["id"]),
        ).rowcount
        report.cancelled_model_calls += db.execute(
            "UPDATE model_calls SET status = 'cancelled', ended_at = ? WHERE attempt_id = ? AND status = 'running'",
            (now, attempt["id"]),
        ).rowcount
        rt.events.emit(
            "recovery.attempt_interrupted",
            project_id=task.project_id,
            task_id=task_id,
            attempt_id=attempt["id"],
            level="warning",
            data={"node": checkpoint_node, "previous_status": task.status.value},
        )
        if task.status in ACTIVE or task.status in {TaskStatus.AWAITING_APPROVAL, TaskStatus.NEEDS_INPUT}:
            reason = (
                "interrupted during finalization; workspace/apply state must be inspected"
                if ambiguous
                else f"runtime stopped during '{checkpoint_node}'; resumable from checkpoint"
            )
            try:
                rt.tasks.transition(task_id, TaskStatus.INTERRUPTED, reason=reason, actor="recovery")
                if ambiguous:
                    rt.tasks.transition(task_id, TaskStatus.UNKNOWN, reason=reason, actor="recovery")
            except CoreError as exc:
                report.notes.append(f"{task_id}: {exc.message}")
            (report.unknown_tasks if ambiguous else report.interrupted_tasks).append(entry)
    for proc in db.query("SELECT * FROM processes WHERE status = 'running'"):
        owner_dead = proc["runtime_id"] in dead_runtimes or proc["runtime_id"] is None
        if proc["runtime_id"] == rt.runtime_id or not owner_dead:
            continue
        alive = process_alive(int(proc["pid"]), proc["start_ticks"])
        info = {"pid": proc["pid"], "command": proc["command"][:200], "task": proc["task_id"], "alive": alive}
        if not alive:
            if not dry_run:
                db.execute(
                    "UPDATE processes SET status = 'reaped', ended_at = ? WHERE id = ?", (now, proc["id"])
                )
            continue
        report.orphan_processes.append(info)
        if kill_orphans and not dry_run:
            pgid = int(proc["pgid"] or proc["pid"])
            kill_group(pgid, signal.SIGTERM)
            try:
                os.kill(int(proc["pid"]), 0)
                kill_group(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            db.execute("UPDATE processes SET status = 'reaped', ended_at = ? WHERE id = ?", (now, proc["id"]))
            report.killed_processes.append(int(proc["pid"]))
        elif not dry_run:
            db.execute("UPDATE processes SET status = 'orphaned' WHERE id = ?", (proc["id"],))
    for ws in rt.workspaces.list(statuses=("active", "preserved")):
        if ws.isolated and not ws.path.exists():
            report.orphaned_workspaces.append(ws.id)
            if not dry_run:
                rt.workspaces.set_status(ws, "orphaned", reason="workspace directory missing")
    if report.changed and not dry_run:
        rt.events.emit(
            "recovery.completed", level="warning", data={k: v for k, v in report.to_dict().items() if v}
        )
    return report
