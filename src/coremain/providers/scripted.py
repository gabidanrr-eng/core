"""Deterministic scripted provider for local development, tests, demos and evaluation harness
validation. It is only used when explicitly configured (``kind = "scripted"``) and every model
call made through it is labelled as scripted; it never stands in for a real model silently.

Script format (JSON)::

    {"name": "demo",
     "roles": {"implementer": [TURN, TURN, ...], "reviewer": [TURN], ...},
     "default": TURN}

The turn index is the number of assistant messages already in the conversation, so replay is
stateless and deterministic (it survives crashes and resumption). A TURN is::

    {"text": "...", "tool_calls": [{"name": "read_file", "arguments": {...}}],
     "error": {"class": "rate_limited", "status": 429, "message": "..."},
     "malformed_arguments": true, "delay_s": 0.5, "usage": {"input_tokens": 10, "output_tokens": 5}}

Strings may contain ``{{last_tool_output}}`` which is replaced with the most recent tool result.

An ``error`` with ``"times": N`` fails only the first N requests for that turn (per process) and
then serves the turn's text/tool calls, which models transient failures that succeed on retry.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from coremain.config.schema import ProviderConfig
from coremain.providers.base import Provider
from coremain.providers.errors import ProviderError, ProviderErrorClass
from coremain.providers.types import (
    ChatRequest,
    DiscoveredModel,
    Finish,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    Usage,
    UsageUpdate,
)
from coremain.util.text import estimate_tokens


class ScriptedProvider(Provider):
    kind = "scripted"
    requires_credential = False

    def __init__(
        self,
        provider_id: str,
        config: ProviderConfig,
        credential: str | None,
        secret_headers: dict[str, str] | None = None,
        **kw: Any,
    ):
        super().__init__(provider_id, config, credential, secret_headers, **kw)
        self._script: dict[str, Any] | None = None
        self._fired: Counter[str] = Counter()

    def script(self) -> dict[str, Any]:
        if self._script is None:
            path = Path(self.config.script or "").expanduser()
            try:
                self._script = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    f"{self.id}: cannot load script {path}: {exc}",
                    error_class=ProviderErrorClass.NOT_CONFIGURED,
                    provider_id=self.id,
                ) from exc
        return self._script

    def _turn(self, req: ChatRequest) -> tuple[str, dict[str, Any]]:
        script = self.script()
        role = str(req.metadata.get("role") or "default")
        node = str(req.metadata.get("node") or "")
        index = sum(1 for m in req.messages if m.role == "assistant")
        key = f"{role}@{node}#{index}"
        roles = script.get("roles", {})
        turns = roles.get(f"{role}@{node}") or roles.get(role)
        if isinstance(turns, list) and index < len(turns):
            return key, dict(turns[index])
        default = script.get("default")
        if isinstance(default, dict):
            return key, dict(default)
        return key, {
            "text": f"[scripted provider '{script.get('name', self.id)}' has no turn {index} for role {role}]"
        }

    @staticmethod
    def _substitute(value: Any, last_output: str) -> Any:
        if isinstance(value, str):
            return value.replace("{{last_tool_output}}", last_output)
        if isinstance(value, dict):
            return {k: ScriptedProvider._substitute(v, last_output) for k, v in value.items()}
        if isinstance(value, list):
            return [ScriptedProvider._substitute(v, last_output) for v in value]
        return value

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        key, turn = self._turn(req)
        last_output = next((m.content for m in reversed(req.messages) if m.role == "tool"), "")
        turn = self._substitute(turn, last_output)
        if turn.get("delay_s"):
            await asyncio.sleep(float(turn["delay_s"]))
        err = turn.get("error")
        if isinstance(err, dict) and (err.get("times") is None or self._fired[key] < int(err["times"])):
            self._fired[key] += 1
            raise ProviderError(
                f"{self.id}: scripted {err.get('class', 'server_error')}: {err.get('message', 'injected failure')}",
                error_class=ProviderErrorClass(err.get("class", "server_error")),
                provider_id=self.id,
                model_id=req.model,
                status_code=err.get("status"),
                retry_after=err.get("retry_after"),
            )
        text = str(turn.get("text", ""))
        for i in range(0, len(text), 40):
            yield TextDelta(text[i : i + 40])
        calls = turn.get("tool_calls") or []
        for i, call in enumerate(calls):
            yield ToolCallStart(i, call.get("id") or f"script_{i}", call["name"])
            raw = "{not json" if turn.get("malformed_arguments") else json.dumps(call.get("arguments", {}))
            yield ToolCallDelta(i, raw)
        usage = turn.get("usage") or {}
        prompt_chars = sum(len(m.content) for m in req.messages)
        yield UsageUpdate(
            Usage(
                input_tokens=int(usage.get("input_tokens", estimate_tokens("x" * prompt_chars))),
                output_tokens=int(usage.get("output_tokens", estimate_tokens(text) + 8 * len(calls))),
            )
        )
        yield Finish("tool_calls" if calls else "stop")

    async def list_models(self) -> list[DiscoveredModel]:
        return [
            DiscoveredModel(id=str(m), display_name=f"scripted:{m}")
            for m in self.script().get("models", ["scripted"])
        ]
