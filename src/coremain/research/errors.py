"""Typed research failures with stable ``error_class`` values and deterministic exit codes."""

from __future__ import annotations

from typing import Any

from coremain.errors import CoreError, ExitCode

# error_class -> exit code; anything not listed (auth_failed, rate_limited, upstream_error, not_ready,
# network_error, timeout, invalid_response, credential_missing) means the capability is blocked.
_EXIT_CODES: dict[str, ExitCode] = {
    "not_found": ExitCode.NOT_FOUND,
    "invalid_url": ExitCode.USAGE,
    "invalid_request": ExitCode.USAGE,
    "http_error": ExitCode.FAILURE,
    "unsupported_content": ExitCode.FAILURE,
    "redirect_limit": ExitCode.FAILURE,
}


class ResearchError(CoreError):
    """A documentation lookup or web fetch failed. ``error_class`` is a stable classification."""

    code = "research_error"
    exit_code = ExitCode.BLOCKED

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        status: int | None = None,
        retry_after_s: int | None = None,
        **kw: Any,
    ):
        super().__init__(message, **kw)
        self.error_class = error_class
        self.status = status
        self.retry_after_s = retry_after_s
        self.exit_code = _EXIT_CODES.get(error_class, ExitCode.BLOCKED)
        if status is not None:
            self.details.setdefault("status", status)
        if retry_after_s is not None:
            self.details.setdefault("retry_after_s", retry_after_s)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["error_class"] = self.error_class
        return out
