"""Typed error taxonomy shared across Core Main.

Errors carry a stable machine-readable ``code`` and a deterministic process exit code so the
CLI can map failures to automation-friendly results without string matching.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class ExitCode(IntEnum):
    OK = 0
    FAILURE = 1
    USAGE = 2
    INCOMPLETE = 3  # work finished but evidence does not prove completion / needs attention
    BLOCKED = 4  # blocked on credentials, billing, provider or integration availability
    POLICY_DENIED = 5
    NOT_FOUND = 6
    CONFLICT = 7
    CANCELLED = 8
    CONFIG = 9
    INTERNAL = 10


class CoreError(Exception):
    code = "core_error"
    exit_code: ExitCode = ExitCode.FAILURE

    def __init__(self, message: str, *, hint: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"class": self.code, "message": self.message}
        if self.hint:
            out["hint"] = self.hint
        if self.details:
            out["details"] = self.details
        return out

    def __str__(self) -> str:
        return self.message


class ConfigError(CoreError):
    code = "config_error"
    exit_code = ExitCode.CONFIG


class UsageError(CoreError):
    code = "usage_error"
    exit_code = ExitCode.USAGE


class NotFoundError(CoreError):
    code = "not_found"
    exit_code = ExitCode.NOT_FOUND


class ConflictError(CoreError):
    code = "conflict"
    exit_code = ExitCode.CONFLICT


class InvalidTransitionError(ConflictError):
    code = "invalid_transition"


class LeaseLostError(ConflictError):
    """Raised when a worker's fencing token no longer owns the resource it is mutating."""

    code = "lease_lost"


class GateFailedError(ConflictError):
    """Completion was requested but the evidence gate did not pass."""

    code = "evidence_gate_failed"


class PolicyDeniedError(CoreError):
    code = "policy_denied"
    exit_code = ExitCode.POLICY_DENIED


class BudgetExceededError(CoreError):
    code = "budget_exceeded"


class OperationCancelled(CoreError):
    code = "cancelled"
    exit_code = ExitCode.CANCELLED


class IntegrityFailure(CoreError):
    code = "integrity_error"


class MigrationError(CoreError):
    code = "migration_error"
    exit_code = ExitCode.CONFIG


class WorkspaceError(CoreError):
    code = "workspace_error"


class WorkspaceConflictError(WorkspaceError):
    code = "workspace_conflict"
    exit_code = ExitCode.CONFLICT


class ToolError(CoreError):
    """A tool failed. ``error_class`` is a stable classification (timeout, not_found, ...)."""

    code = "tool_error"

    def __init__(self, message: str, *, error_class: str = "tool_failed", **kw: Any):
        super().__init__(message, **kw)
        self.error_class = error_class

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["error_class"] = self.error_class
        return out


class MCPError(CoreError):
    code = "mcp_error"
    exit_code = ExitCode.BLOCKED

    def __init__(self, message: str, *, error_class: str = "mcp_failed", server: str | None = None, **kw: Any):
        super().__init__(message, **kw)
        self.error_class = error_class
        self.server = server

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["error_class"] = self.error_class
        if self.server:
            out["server"] = self.server
        return out


class CapabilityUnavailable(CoreError):
    """An optional capability (browser, LSP, docs retrieval, ...) is not available right now."""

    code = "capability_unavailable"
    exit_code = ExitCode.BLOCKED
