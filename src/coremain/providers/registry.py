"""Provider and model registry.

Models are only ever the ones the user configured (by alias → provider + exact model id).
Providers are instantiated lazily; credentials are resolved at first use, registered with the
redactor and never persisted.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from coremain.config.schema import CoreConfig, ModelConfig
from coremain.errors import NotFoundError
from coremain.providers.anthropic import AnthropicProvider
from coremain.providers.base import Provider
from coremain.providers.errors import ProviderError, ProviderErrorClass
from coremain.providers.openai_compat import OpenAICompatibleProvider
from coremain.providers.scripted import ScriptedProvider
from coremain.security.credentials import CredentialStore, ResolvedCredential, resolve_credential
from coremain.security.redact import Redactor

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "scripted": ScriptedProvider,
}

# Endpoint presets for `core providers add`. These describe *providers* only; model ids are
# never preset — they come from the user or from live discovery.
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "kind": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key": "env:OPENAI_API_KEY",
    },
    "anthropic": {
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "env:ANTHROPIC_API_KEY",
    },
    "openrouter": {
        "kind": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "env:OPENROUTER_API_KEY",
    },
    "groq": {
        "kind": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": "env:GROQ_API_KEY",
    },
    "deepseek": {
        "kind": "openai_compatible",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "env:DEEPSEEK_API_KEY",
    },
    "mistral": {
        "kind": "openai_compatible",
        "base_url": "https://api.mistral.ai/v1",
        "api_key": "env:MISTRAL_API_KEY",
    },
    "xai": {"kind": "openai_compatible", "base_url": "https://api.x.ai/v1", "api_key": "env:XAI_API_KEY"},
    "together": {
        "kind": "openai_compatible",
        "base_url": "https://api.together.xyz/v1",
        "api_key": "env:TOGETHER_API_KEY",
    },
    "gemini": {
        "kind": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key": "env:GEMINI_API_KEY",
    },
    "ollama": {"kind": "openai_compatible", "base_url": "http://localhost:11434/v1"},
    "lmstudio": {"kind": "openai_compatible", "base_url": "http://localhost:1234/v1"},
}


@dataclass(frozen=True)
class ModelProfile:
    key: str
    provider_id: str
    model_id: str
    config: ModelConfig

    @property
    def ref(self) -> str:
        return f"{self.provider_id}/{self.model_id}"

    def strength(self, dim: str) -> float:
        if dim in self.config.strengths:
            return self.config.strengths[dim]
        return {"frontier": 0.9, "strong": 0.78, "standard": 0.62, "light": 0.45}[self.config.tier]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "ref": self.ref,
            "provider": self.provider_id,
            "model": self.model_id,
            **self.config.model_dump(mode="json", exclude={"provider", "id"}),
        }


class ProviderRegistry:
    def __init__(
        self,
        config: CoreConfig,
        store: CredentialStore,
        redactor: Redactor,
        *,
        env: Mapping[str, str] | None = None,
        transports: dict[str, httpx.AsyncBaseTransport] | None = None,
    ):
        self.config = config
        self.store = store
        self.redactor = redactor
        self.env = os.environ if env is None else env
        self._providers: dict[str, Provider] = {}
        self._transports = transports or {}

    def provider_ids(self) -> list[str]:
        return sorted(self.config.providers)

    def credential_status(self, provider_id: str) -> ResolvedCredential:
        cfg = self.config.providers[provider_id]
        if cfg.kind == "scripted":
            return ResolvedCredential("", "scripted", "none")
        if cfg.api_key is None:
            return ResolvedCredential("", None, "none", "no api_key configured (allowed for local servers)")
        return resolve_credential(cfg.api_key, self.store, env=self.env, redactor=self.redactor)

    def provider(self, provider_id: str) -> Provider:
        if provider_id in self._providers:
            return self._providers[provider_id]
        cfg = self.config.providers.get(provider_id)
        if cfg is None:
            raise ProviderError(
                f"provider '{provider_id}' is not configured",
                error_class=ProviderErrorClass.NOT_CONFIGURED,
                provider_id=provider_id,
            )
        if not cfg.enabled:
            raise ProviderError(
                f"provider '{provider_id}' is disabled",
                error_class=ProviderErrorClass.NOT_CONFIGURED,
                provider_id=provider_id,
            )
        credential = None
        if cfg.api_key:
            credential = resolve_credential(
                cfg.api_key, self.store, env=self.env, redactor=self.redactor
            ).value
        secret_headers: dict[str, str] = {}
        for header, ref in cfg.secret_headers.items():
            value = resolve_credential(ref, self.store, env=self.env, redactor=self.redactor).value
            if value:
                secret_headers[header] = value
        cls = PROVIDER_CLASSES[cfg.kind]
        provider = cls(
            provider_id, cfg, credential, secret_headers, transport=self._transports.get(provider_id)
        )
        self._providers[provider_id] = provider
        return provider

    def models(self, *, include_disabled: bool = False) -> list[ModelProfile]:
        out = []
        for key, m in self.config.models.items():
            provider_cfg = self.config.providers.get(m.provider)
            if not include_disabled and (not m.enabled or provider_cfg is None or not provider_cfg.enabled):
                continue
            out.append(ModelProfile(key, m.provider, m.id, m))
        return sorted(out, key=lambda p: p.key)

    def model(self, key_or_ref: str) -> ModelProfile:
        for profile in self.models(include_disabled=True):
            if key_or_ref in (profile.key, profile.ref):
                return profile
        raise NotFoundError(
            f"model '{key_or_ref}' is not configured",
            hint="Configured models: "
            + (", ".join(p.key for p in self.models()) or "none")
            + ". Add one with `core models add`.",
        )

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        self._providers.clear()
