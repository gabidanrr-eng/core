"""Provider interface and shared HTTP plumbing."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from coremain.config.schema import ProviderConfig
from coremain.providers.classify import classify_http, classify_transport
from coremain.providers.errors import ProviderError, ProviderErrorClass
from coremain.providers.types import ChatRequest, DiscoveredModel, StreamEvent
from coremain.version import __version__


class Provider(ABC):
    kind: str = "abstract"
    requires_credential: bool = True

    def __init__(self, provider_id: str, config: ProviderConfig, credential: str | None, secret_headers: dict[str, str] | None = None,
                 *, transport: httpx.AsyncBaseTransport | None = None):
        self.id = provider_id
        self.config = config
        self.credential = credential
        self.secret_headers = secret_headers or {}
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(connect=self.config.connect_timeout_s, read=self.config.read_timeout_s, write=60.0, pool=30.0)
            self._client = httpx.AsyncClient(
                timeout=timeout, transport=self._transport, headers={"user-agent": f"core-main/{__version__}"},
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _check_credential(self) -> None:
        if self.requires_credential and not self.credential:
            raise ProviderError(
                f"{self.id}: no credential available ({self.config.api_key or 'api_key not configured'})",
                error_class=ProviderErrorClass.AUTH_FAILED, provider_id=self.id,
                hint=f"Set the referenced credential or run `core providers login {self.id}`.",
            )

    @asynccontextmanager
    async def _post_stream(self, url: str, body: dict[str, Any], headers: dict[str, str], model_id: str | None) -> AsyncIterator[httpx.Response]:
        try:
            async with self.client.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code >= 400:
                    text = (await response.aread()).decode("utf-8", errors="replace")
                    raise classify_http(response.status_code, text, response.headers, provider_id=self.id, model_id=model_id)
                yield response
        except httpx.HTTPError as exc:
            raise classify_transport(exc, provider_id=self.id, model_id=model_id) from exc

    async def _get_json(self, url: str, headers: dict[str, str]) -> Any:
        try:
            response = await self.client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise classify_transport(exc, provider_id=self.id, model_id=None) from exc
        if response.status_code >= 400:
            raise classify_http(response.status_code, response.text, response.headers, provider_id=self.id, model_id=None)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.id}: model listing returned non-JSON", error_class=ProviderErrorClass.MALFORMED_RESPONSE,
                                provider_id=self.id) from exc

    @abstractmethod
    def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion. Raises ProviderError with an exact class on failure."""

    @abstractmethod
    async def list_models(self) -> list[DiscoveredModel]:
        """List models the credential can access (live request)."""


def parse_json_arguments(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    if not raw.strip():
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} at position {exc.pos}"
    if not isinstance(value, dict):
        return None, "arguments must be a JSON object"
    return value, None
