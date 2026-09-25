"""Provider error taxonomy.

Provider failures keep their exact class all the way to the user. They are never converted
into assistant text, never silently retried beyond the configured bound and never trigger a
fallback unless the user configured one explicitly.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from coremain.errors import CoreError, ExitCode


class ProviderErrorClass(StrEnum):
    AUTH_FAILED = "auth_failed"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    RATE_LIMITED = "rate_limited"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_OVERFLOW = "context_overflow"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    OVERLOADED = "overloaded"
    CONTENT_FILTERED = "content_filtered"
    MALFORMED_RESPONSE = "malformed_response"
    NOT_CONFIGURED = "not_configured"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


TRANSIENT = frozenset(
    {
        ProviderErrorClass.RATE_LIMITED,
        ProviderErrorClass.NETWORK_ERROR,
        ProviderErrorClass.TIMEOUT,
        ProviderErrorClass.SERVER_ERROR,
        ProviderErrorClass.OVERLOADED,
    }
)

# Classes that block the task until a human acts (credentials, billing, configuration).
BLOCKING = {
    ProviderErrorClass.AUTH_FAILED: "credentials",
    ProviderErrorClass.INSUFFICIENT_FUNDS: "billing",
    ProviderErrorClass.MODEL_UNAVAILABLE: "provider_unavailable",
    ProviderErrorClass.NOT_CONFIGURED: "configuration",
}

HINTS = {
    ProviderErrorClass.AUTH_FAILED: "Check the credential with `core providers check {provider}`; update it with `core providers login {provider}`.",
    ProviderErrorClass.INSUFFICIENT_FUNDS: "The provider reports insufficient credit/quota. Top up the account or choose another configured model.",
    ProviderErrorClass.RATE_LIMITED: "The provider is rate limiting requests. Core Main retries within the configured bound; consider lowering max_workers.",
    ProviderErrorClass.MODEL_UNAVAILABLE: "The model id is not available to this account. Run `core models discover {provider}` to list accessible models.",
    ProviderErrorClass.INVALID_REQUEST: "The provider rejected the request. This usually indicates an unsupported parameter for this model.",
    ProviderErrorClass.CONTEXT_OVERFLOW: "The compiled context exceeded the model's window. Lower context budgets or pick a larger-context model.",
    ProviderErrorClass.NETWORK_ERROR: "Could not reach the provider. Check connectivity, proxies and base_url.",
    ProviderErrorClass.TIMEOUT: "The provider did not respond in time. Increase read_timeout_s for slow reasoning models.",
    ProviderErrorClass.SERVER_ERROR: "The provider returned an internal error.",
    ProviderErrorClass.OVERLOADED: "The provider is overloaded. Retry later or configure an explicit fallback.",
    ProviderErrorClass.CONTENT_FILTERED: "The provider refused or filtered the content.",
    ProviderErrorClass.MALFORMED_RESPONSE: "The provider/model returned a response Core Main could not parse.",
    ProviderErrorClass.NOT_CONFIGURED: "Configure a provider and model: `core providers add` then `core models add`.",
}


class ProviderError(CoreError):
    code = "provider_error"

    def __init__(
        self,
        message: str,
        *,
        error_class: ProviderErrorClass,
        provider_id: str | None = None,
        model_id: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        request_id: str | None = None,
        raw_code: str | None = None,
        hint: str | None = None,
    ):
        if hint is None and error_class in HINTS:
            hint = HINTS[error_class].format(provider=provider_id or "<provider>")
        super().__init__(message, hint=hint)
        self.error_class = error_class
        self.provider_id = provider_id
        self.model_id = model_id
        self.status_code = status_code
        self.retry_after = retry_after
        self.request_id = request_id
        self.raw_code = raw_code
        self.exit_code = ExitCode.BLOCKED if error_class in BLOCKING else ExitCode.FAILURE

    @property
    def transient(self) -> bool:
        return self.error_class in TRANSIENT

    @property
    def block_reason(self) -> str | None:
        return BLOCKING.get(self.error_class)

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out.update(
            {
                "error_class": self.error_class.value,
                "provider": self.provider_id,
                "model": self.model_id,
                "status_code": self.status_code,
                "retry_after": self.retry_after,
                "request_id": self.request_id,
                "raw_code": self.raw_code,
            }
        )
        return {k: v for k, v in out.items() if v is not None}
