"""Entity records materialized from the database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from coremain.domain.states import AttemptStatus, TaskStatus
from coremain.util.jsonutil import loads


@dataclass
class Project:
    id: str
    root_path: str
    name: str
    vcs: str
    git_remote: str | None
    profile: dict[str, Any] | None
    profile_hash: str | None
    trusted_config_hash: str | None
    created_at: float

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Project:
        return cls(r["id"], r["root_path"], r["name"], r["vcs"], r["git_remote"], loads(r["profile_json"]),
                   r["profile_hash"], r["trusted_config_hash"], r["created_at"])


@dataclass
class Session:
    id: str
    project_id: str
    title: str
    status: str
    parent_session_id: str | None
    fork_checkpoint_id: str | None
    intent: str | None
    summary: str | None
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Session:
        return cls(r["id"], r["project_id"], r["title"], r["status"], r["parent_session_id"], r["fork_checkpoint_id"],
                   r["intent"], r["summary"], r["created_at"], r["updated_at"])


@dataclass
class Message:
    id: str
    session_id: str
    task_id: str | None
    role: str
    content: str
    meta: dict[str, Any]
    created_at: float

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Message:
        return cls(r["id"], r["session_id"], r["task_id"], r["role"], r["content"], loads(r["meta_json"], {}),
                   r["created_at"])


@dataclass
class Task:
    id: str
    project_id: str
    session_id: str | None
    parent_task_id: str | None
    kind: str
    title: str
    description: str
    status: TaskStatus
    status_reason: str | None
    block_reason: str | None
    mode: str | None
    decision: dict[str, Any]
    contract: dict[str, Any]
    options: dict[str, Any]
    priority: int
    depth: int
    workspace_id: str | None
    result_summary: str | None
    evidence_level: str | None
    attempt_count: int
    recovered_count: int
    cancel_requested_at: float | None
    cancel_reason: str | None
    pause_requested_at: float | None
    version: int
    created_at: float
    updated_at: float
    completed_at: float | None
    idempotency_key: str | None = None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Task:
        return cls(
            id=r["id"], project_id=r["project_id"], session_id=r["session_id"], parent_task_id=r["parent_task_id"],
            kind=r["kind"], title=r["title"], description=r["description"], status=TaskStatus(r["status"]),
            status_reason=r["status_reason"], block_reason=r["block_reason"], mode=r["mode"],
            decision=loads(r["decision_json"], {}), contract=loads(r["contract_json"], {}),
            options=loads(r["options_json"], {}), priority=r["priority"], depth=r["depth"],
            workspace_id=r["workspace_id"], result_summary=r["result_summary"], evidence_level=r["evidence_level"],
            attempt_count=r["attempt_count"], recovered_count=r["recovered_count"],
            cancel_requested_at=r["cancel_requested_at"], cancel_reason=r["cancel_reason"],
            pause_requested_at=r["pause_requested_at"], version=r["version"], created_at=r["created_at"],
            updated_at=r["updated_at"], completed_at=r["completed_at"], idempotency_key=r["idempotency_key"],
        )

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["status"] = self.status.value
        return d


@dataclass
class Attempt:
    id: str
    task_id: str
    number: int
    status: AttemptStatus
    runtime_id: str | None
    fence_token: int
    workspace_id: str | None
    workflow: str | None
    checkpoint: dict[str, Any]
    base_fingerprint: str | None
    resumed_from: str | None
    error_class: str | None
    error_message: str | None
    usage: dict[str, Any]
    started_at: float
    heartbeat_at: float | None
    ended_at: float | None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Attempt:
        return cls(r["id"], r["task_id"], r["number"], AttemptStatus(r["status"]), r["runtime_id"], r["fence_token"],
                   r["workspace_id"], r["workflow"], loads(r["checkpoint_json"], {}), r["base_fingerprint"],
                   r["resumed_from"], r["error_class"], r["error_message"], loads(r["usage_json"], {}),
                   r["started_at"], r["heartbeat_at"], r["ended_at"])

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["status"] = self.status.value
        return d


@dataclass
class GateResult:
    passed: bool
    summary: str
    level: str = "none"
    satisfied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failing: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
