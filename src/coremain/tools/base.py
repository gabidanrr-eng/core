"""Tool contracts.

A tool is a typed capability: a pydantic input model (validated before execution), a policy
capability and side-effect class, a timeout, and an async ``run``. Tools never evaluate
policy themselves; ``ToolExecutor`` does that uniformly for native and MCP tools.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from coremain.config.schema import CoreConfig
from coremain.runtime.cancel import CancelToken
from coremain.runtime.leases import Fence
from coremain.security.policy import Capability, PolicyRequest

if TYPE_CHECKING:
    from coremain.events import EventLog
    from coremain.exec.process import ProcessRunner
    from coremain.runtime.approvals import Approval, ApprovalService
    from coremain.runtime.tasks import TaskService
    from coremain.security.policy import PolicyEngine
    from coremain.security.redact import Redactor
    from coremain.store.artifacts import ArtifactStore
    from coremain.store.db import Database
    from coremain.verify.evidence import EvidenceEngine
    from coremain.workspaces.manager import Workspace, WorkspaceManager


class SideEffect(StrEnum):
    NONE = "none"
    LOCAL = "local"
    DESTRUCTIVE = "destructive"
    EXTERNAL = "external"


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class ApprovalDecision:
    approved: bool
    scope: str = "once"
    reason: str | None = None


ApprovalHandler = Callable[["Approval"], Awaitable[ApprovalDecision]]
InputHandler = Callable[[str, str | None], Awaitable[str]]


@dataclass
class ToolServices:
    """Runtime services available to tools. Optional subsystems are ``None`` when unavailable."""

    config: CoreConfig
    db: Database
    policy: PolicyEngine
    approvals: ApprovalService
    tasks: TaskService
    processes: ProcessRunner
    artifacts: ArtifactStore
    evidence: EvidenceEngine
    events: EventLog
    redactor: Redactor
    workspaces: WorkspaceManager
    approval_handler: ApprovalHandler | None = None
    input_handler: InputHandler | None = None
    intel: Any = None
    memory: Any = None
    skills: Any = None
    mcp: Any = None
    browser: Any = None
    research: Any = None
    lsp: Any = None
    profile: dict[str, Any] = field(default_factory=dict)
    spawn_subtask: Callable[..., Awaitable[str]] | None = None
    # Base environment for child processes (the runtime's env, not necessarily os.environ).
    base_env: dict[str, str] | None = None


@dataclass
class ToolContext:
    project_id: str
    session_id: str | None
    task_id: str | None
    attempt_id: str | None
    workspace: Workspace
    fence: Fence | None
    cancel: CancelToken
    role: str
    services: ToolServices
    node: str | None = None
    model_call_id: str | None = None
    read_hashes: dict[str, str] = field(default_factory=dict)
    files_read: set[str] = field(default_factory=set)
    changed_paths: set[str] = field(default_factory=set)
    commands_run: list[dict[str, Any]] = field(default_factory=list)
    structured: dict[str, Any] = field(default_factory=dict)
    loaded_skills: set[str] = field(default_factory=set)
    extra_env: dict[str, str] = field(default_factory=dict)
    depth: int = 0

    @property
    def root(self) -> Any:
        return self.workspace.path


@dataclass
class ToolResult:
    ok: bool
    content: str
    status: str = "ok"
    error_class: str | None = None
    data: dict[str, Any] | None = None
    artifact_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    images: list[tuple[str, str]] = field(default_factory=list)
    terminal: bool = False
    payload: dict[str, Any] | None = None

    @classmethod
    def error(cls, message: str, error_class: str = "tool_error", status: str = "error") -> ToolResult:
        return cls(False, message, status=status, error_class=error_class)


def inline_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local ``$ref``s into a self-contained schema and drop noisy ``title`` keys."""
    defs = schema.get("$defs", {})

    def walk(node: Any, depth: int = 0) -> Any:
        if depth > 30:
            return node
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str) and node["$ref"].startswith("#/$defs/"):
                target = copy.deepcopy(defs.get(node["$ref"].split("/")[-1], {}))
                merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
                return walk(merged, depth + 1)
            return {k: walk(v, depth + 1) for k, v in node.items() if k not in {"$defs", "title"}}
        if isinstance(node, list):
            return [walk(v, depth + 1) for v in node]
        return node

    out = walk(schema)
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    Input: ClassVar[type[BaseModel]]
    capability: ClassVar[str] = Capability.FS_READ
    side_effect: ClassVar[SideEffect] = SideEffect.NONE
    timeout_s: ClassVar[float] = 60.0
    terminal: ClassVar[bool] = False
    read_only: ClassVar[bool] = True

    def target(self, args: Any) -> str:
        return ""

    def summarize(self, args: Any) -> str:
        target = self.target(args)
        return f"{self.name} {target}".strip()

    def policy_request(self, ctx: ToolContext, args: Any) -> PolicyRequest | None:
        return PolicyRequest(
            capability=self.capability,
            target=self.target(args),
            workspace_root=ctx.workspace.path,
            workspace_isolated=ctx.workspace.isolated,
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            tool=self.name,
        )

    def effective_timeout(self, args: Any) -> float:
        return self.timeout_s

    async def run(self, ctx: ToolContext, args: Any) -> ToolResult:
        raise NotImplementedError

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return inline_schema(cls.Input.model_json_schema())

    @classmethod
    def spec(cls) -> dict[str, Any]:
        return {"name": cls.name, "description": cls.description, "parameters": cls.schema()}
