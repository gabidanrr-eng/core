"""Map HTTP/transport failures to exact provider error classes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import httpx

from coremain.providers.errors import ProviderError, ProviderErrorClass

_CONTEXT = re.compile(
    r"(context[_ ]length|maximum context|context window|too many tokens|prompt is too long|input is too long|"
    r"reduce the length|exceeds the (?:model'?s )?(?:context|maximum)|token limit)", re.I
)
_FUNDS = re.compile(r"(insufficient[_ ]quota|credit balance is too low|insufficient (?:funds|credits|balance)|"
                    r"exceeded your current quota|billing|payment required|out of credits)", re.I)
_MODEL = re.compile(r"(model[^.]{0,60}(?:not found|does not exist|not available|unavailable|not supported|invalid)|"
                    r"(?:no such|unknown|invalid) model|model_not_found)", re.I)
_FILTER = re.compile(r"(content[_ ]policy|content[_ ]filter|safety|flagged|moderation)", re.I)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    for key in ("retry-after-ms", "retry-after"):
        value = headers.get(key)
        if value:
            try:
                seconds = float(value)
                return seconds / 1000 if key.endswith("ms") else seconds
            except ValueError:
                continue
    return None


def _extract(body: Any) -> tuple[str, str | None, str | None]:
    """Return (message, code, type) from common provider error envelopes."""
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            message = str(err.get("message") or err.get("detail") or body.get("message") or "")
            code = err.get("code")
            etype = err.get("type")
            meta = err.get("metadata")
            if isinstance(meta, dict) and meta.get("raw"):
                message = f"{message} ({str(meta['raw'])[:300]})"
            return message, str(code) if code is not None else None, str(etype) if etype is not None else None
        if isinstance(err, str):
            return err, None, None
    return (str(body)[:500] if body else ""), None, None


def classify_http(status: int, body_text: str, headers: Mapping[str, str], *, provider_id: str, model_id: str | None) -> ProviderError:
    try:
        body: Any = json.loads(body_text) if body_text else None
    except json.JSONDecodeError:
        body = body_text
    message, code, etype = _extract(body)
    blob = " ".join(x for x in (message, code or "", etype or "") if x)
    request_id = headers.get("x-request-id") or headers.get("request-id") or headers.get("cf-ray")
    ec: ProviderErrorClass
    if status in (401,) or etype == "authentication_error":
        ec = ProviderErrorClass.AUTH_FAILED
    elif status == 402 or _FUNDS.search(blob) or etype == "billing_error":
        ec = ProviderErrorClass.INSUFFICIENT_FUNDS
    elif status == 403 or etype == "permission_error":
        ec = ProviderErrorClass.AUTH_FAILED
    elif status == 429:
        ec = ProviderErrorClass.RATE_LIMITED
    elif status == 413 or etype == "request_too_large" or _CONTEXT.search(blob):
        ec = ProviderErrorClass.CONTEXT_OVERFLOW
    elif status == 404:
        ec = ProviderErrorClass.MODEL_UNAVAILABLE if (_MODEL.search(blob) or "model" in blob.lower()) else ProviderErrorClass.INVALID_REQUEST
    elif status in (400, 422):
        if _MODEL.search(blob):
            ec = ProviderErrorClass.MODEL_UNAVAILABLE
        elif _FILTER.search(blob) and "policy" in blob.lower():
            ec = ProviderErrorClass.CONTENT_FILTERED
        else:
            ec = ProviderErrorClass.INVALID_REQUEST
    elif status == 408:
        ec = ProviderErrorClass.TIMEOUT
    elif status in (503, 529) or etype == "overloaded_error":
        ec = ProviderErrorClass.OVERLOADED
    elif status >= 500:
        ec = ProviderErrorClass.SERVER_ERROR
    else:
        ec = ProviderErrorClass.UNKNOWN
    detail = message or f"HTTP {status}"
    return ProviderError(
        f"{provider_id}: {ec.value.replace('_', ' ')} (HTTP {status}): {detail[:400]}", error_class=ec, provider_id=provider_id,
        model_id=model_id, status_code=status, retry_after=_retry_after(headers), request_id=request_id, raw_code=code or etype,
    )


def classify_stream_error(payload: Any, *, provider_id: str, model_id: str | None) -> ProviderError:
    message, code, etype = _extract(payload)
    blob = f"{message} {code or ''} {etype or ''}"
    mapping = {
        "overloaded_error": ProviderErrorClass.OVERLOADED, "rate_limit_error": ProviderErrorClass.RATE_LIMITED,
        "api_error": ProviderErrorClass.SERVER_ERROR, "authentication_error": ProviderErrorClass.AUTH_FAILED,
        "permission_error": ProviderErrorClass.AUTH_FAILED, "invalid_request_error": ProviderErrorClass.INVALID_REQUEST,
        "not_found_error": ProviderErrorClass.MODEL_UNAVAILABLE, "billing_error": ProviderErrorClass.INSUFFICIENT_FUNDS,
    }
    ec = mapping.get(etype or "", ProviderErrorClass.SERVER_ERROR)
    if _CONTEXT.search(blob):
        ec = ProviderErrorClass.CONTEXT_OVERFLOW
    elif _FUNDS.search(blob):
        ec = ProviderErrorClass.INSUFFICIENT_FUNDS
    return ProviderError(f"{provider_id}: stream error: {message[:400] or etype}", error_class=ec, provider_id=provider_id,
                         model_id=model_id, raw_code=code or etype)


def classify_transport(exc: Exception, *, provider_id: str, model_id: str | None) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(f"{provider_id}: request timed out ({type(exc).__name__})", error_class=ProviderErrorClass.TIMEOUT,
                             provider_id=provider_id, model_id=model_id)
    return ProviderError(f"{provider_id}: network error: {type(exc).__name__}: {exc}"[:500], error_class=ProviderErrorClass.NETWORK_ERROR,
                         provider_id=provider_id, model_id=model_id)
