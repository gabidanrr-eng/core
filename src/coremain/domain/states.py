"""Task and attempt lifecycles.

The transition table is the single definition of legal task state changes. ``TaskService``
is the only code that mutates ``tasks.status`` and it enforces this table, fencing tokens,
durable cancellation and the evidence gate for ``completed``.
"""

from __future__ import annotations

from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    VERIFYING = "verifying"
    REVIEWING = "reviewing"
    AWAITING_APPROVAL = "awaiting_approval"
    NEEDS_INPUT = "needs_input"
    BLOCKED = "blocked"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class AttemptStatus(StrEnum):
    RUNNING = "running"
    SUSPENDED = "suspended"  # stopped at a durable checkpoint awaiting a human (approval/input/pause)
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"  # owner vanished (crash, lease expiry); state reconciled by recovery


class BlockReason(StrEnum):
    CREDENTIALS = "credentials"
    BILLING = "billing"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CONFIGURATION = "configuration"
    MCP_UNAVAILABLE = "mcp_unavailable"
    PERMISSION = "permission"
    DEPENDENCY = "dependency"
    WORKSPACE_CONFLICT = "workspace_conflict"
    BUDGET = "budget"


T = TaskStatus
TERMINAL: frozenset[TaskStatus] = frozenset({T.COMPLETED, T.FAILED, T.CANCELLED})
ACTIVE: frozenset[TaskStatus] = frozenset({T.RUNNING, T.VERIFYING, T.REVIEWING})
ATTENTION: frozenset[TaskStatus] = frozenset(
    {T.AWAITING_APPROVAL, T.NEEDS_INPUT, T.BLOCKED, T.PAUSED, T.INTERRUPTED, T.INCOMPLETE, T.UNKNOWN}
)
RESUMABLE: frozenset[TaskStatus] = frozenset(
    {
        T.AWAITING_APPROVAL,
        T.NEEDS_INPUT,
        T.BLOCKED,
        T.PAUSED,
        T.INTERRUPTED,
        T.INCOMPLETE,
        T.UNKNOWN,
        T.FAILED,
    }
)
_WORKING_EXITS = {
    T.AWAITING_APPROVAL,
    T.NEEDS_INPUT,
    T.BLOCKED,
    T.PAUSED,
    T.INTERRUPTED,
    T.COMPLETED,
    T.INCOMPLETE,
    T.FAILED,
    T.CANCELLED,
    T.UNKNOWN,
}

TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    T.PENDING: frozenset({T.QUEUED, T.CANCELLED, T.BLOCKED}),
    T.QUEUED: frozenset({T.RUNNING, T.CANCELLED, T.PAUSED, T.BLOCKED, T.PENDING}),
    T.RUNNING: frozenset({T.VERIFYING, T.REVIEWING, T.QUEUED, *_WORKING_EXITS}),
    T.VERIFYING: frozenset({T.RUNNING, T.REVIEWING, *_WORKING_EXITS}),
    T.REVIEWING: frozenset({T.RUNNING, T.VERIFYING, *_WORKING_EXITS}),
    T.AWAITING_APPROVAL: frozenset(
        {T.RUNNING, T.QUEUED, T.CANCELLED, T.FAILED, T.NEEDS_INPUT, T.INTERRUPTED}
    ),
    T.NEEDS_INPUT: frozenset({T.RUNNING, T.QUEUED, T.CANCELLED, T.FAILED, T.INTERRUPTED}),
    T.BLOCKED: frozenset({T.QUEUED, T.CANCELLED, T.FAILED}),
    T.PAUSED: frozenset({T.QUEUED, T.CANCELLED}),
    T.INTERRUPTED: frozenset({T.QUEUED, T.CANCELLED, T.FAILED, T.UNKNOWN}),
    T.INCOMPLETE: frozenset({T.QUEUED, T.CANCELLED, T.FAILED}),
    T.UNKNOWN: frozenset({T.QUEUED, T.CANCELLED, T.FAILED, T.INCOMPLETE}),
    T.FAILED: frozenset({T.QUEUED}),
    T.CANCELLED: frozenset(),
    T.COMPLETED: frozenset(),
}

# Transitions still permitted after cancellation was requested (everything else is refused so
# a late worker cannot turn cancelled work into "completed").
ALLOWED_AFTER_CANCEL: frozenset[TaskStatus] = frozenset({T.CANCELLED, T.INTERRUPTED, T.UNKNOWN, T.FAILED})


def can_transition(src: TaskStatus, dst: TaskStatus) -> bool:
    return dst in TRANSITIONS[src]


STATUS_LABELS: dict[TaskStatus, str] = {
    T.PENDING: "pending",
    T.QUEUED: "queued",
    T.RUNNING: "running",
    T.VERIFYING: "validating",
    T.REVIEWING: "under review",
    T.AWAITING_APPROVAL: "awaiting approval",
    T.NEEDS_INPUT: "needs input",
    T.BLOCKED: "blocked",
    T.PAUSED: "paused",
    T.INTERRUPTED: "interrupted (resumable)",
    T.COMPLETED: "completed",
    T.INCOMPLETE: "incomplete",
    T.FAILED: "failed",
    T.CANCELLED: "cancelled",
    T.UNKNOWN: "unknown",
}
