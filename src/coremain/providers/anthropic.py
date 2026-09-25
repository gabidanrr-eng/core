"""Anthropic Messages API provider (native streaming, tool use, prompt caching, thinking)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from coremain.providers.base import Provider
from coremain.providers.classify import classify_stream_error
from coremain.providers.errors import ProviderError, ProviderErrorClass
from coremain.providers.sse import iter_sse
from coremain.providers.types import (
    ChatMessage,
    ChatRequest,
    DiscoveredModel,
    Finish,
    ProviderStateUpdate,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    Usage,
    UsageUpdate,
)

API_VERSION = "2023-06-01"
_STOP = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop", "refusal": "content_filter",
         "pause_turn": "other"}
_EFFORT_BUDGET = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 24576}


class AnthropicProvider(Provider):
    kind = "anthropic"

    @property
    def base_url(self) -> str:
        return (self.config.base_url or "https://api.anthropic.com").rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", "anthropic-version": API_VERSION, **self.config.headers, **self.secret_headers}
        if self.credential:
            headers["x-api-key"] = self.credential
        return headers

    def convert(self, messages: list[ChatMessage], *, caching: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system_blocks: list[dict[str, Any]] = []
        out: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict[str, Any]]) -> None:
            if not blocks:
                return
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for m in messages:
            if m.role == "system":
                if m.content:
                    system_blocks.append({"type": "text", "text": m.content})
            elif m.role == "user":
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "image", "source": {"type": "base64", "media_type": i.media_type, "data": i.data_b64}} for i in m.images]
                push("user", blocks or [{"type": "text", "text": "(empty)"}])
            elif m.role == "assistant":
                blocks = []
                state = m.provider_state or {}
                if state.get("provider") == self.id:
                    blocks.extend(state.get("thinking", []))
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments or {}})
                push("assistant", blocks or [{"type": "text", "text": "(no content)"}])
            elif m.role == "tool":
                content: list[dict[str, Any]] = [{"type": "text", "text": m.content or "(no output)"}]
                content += [{"type": "image", "source": {"type": "base64", "media_type": i.media_type, "data": i.data_b64}} for i in m.images]
                push("user", [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": content, "is_error": m.is_error}])
        if caching and system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return system_blocks, out

    def build_body(self, req: ChatRequest) -> dict[str, Any]:
        caching = bool(req.metadata.get("prompt_caching"))
        system, messages = self.convert(req.messages, caching=caching)
        body: dict[str, Any] = {"model": req.model, "messages": messages, "max_tokens": req.max_output_tokens or 8192, "stream": True}
        if system:
            body["system"] = system
        if req.tools:
            tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in req.tools]
            if caching:
                tools[-1]["cache_control"] = {"type": "ephemeral"}
            body["tools"] = tools
            if req.tool_choice:
                body["tool_choice"] = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}[req.tool_choice]
        thinking = req.reasoning_effort and req.metadata.get("reasoning") == "extended"
        if thinking:
            budget = _EFFORT_BUDGET.get(str(req.reasoning_effort), 8192)
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["max_tokens"] = max(body["max_tokens"], budget + 4096)
        elif req.temperature is not None:
            body["temperature"] = req.temperature
        body.update(self.config.extra_body)
        body.update(req.extra)
        return body

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        self._check_credential()
        body = self.build_body(req)
        block_types: dict[int, str] = {}
        thinking_blocks: dict[int, dict[str, Any]] = {}
        tool_index: dict[int, int] = {}
        usage = Usage()
        async with self._post_stream(f"{self.base_url}/v1/messages", body, self._headers(), req.model) as response:
            async for sse in iter_sse(response.aiter_lines()):
                if not sse.data:
                    continue
                try:
                    payload = json.loads(sse.data)
                except json.JSONDecodeError as exc:
                    raise ProviderError(f"{self.id}: malformed stream event", error_class=ProviderErrorClass.MALFORMED_RESPONSE,
                                        provider_id=self.id, model_id=req.model) from exc
                etype = payload.get("type") or sse.event
                if etype == "error":
                    raise classify_stream_error(payload, provider_id=self.id, model_id=req.model)
                if etype == "message_start":
                    u = (payload.get("message") or {}).get("usage") or {}
                    usage.input_tokens = int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) + int(
                        u.get("cache_creation_input_tokens") or 0)
                    usage.cached_tokens = int(u.get("cache_read_input_tokens") or 0)
                    usage.cache_write_tokens = int(u.get("cache_creation_input_tokens") or 0)
                elif etype == "content_block_start":
                    idx = int(payload.get("index", 0))
                    block = payload.get("content_block") or {}
                    btype = block.get("type", "")
                    block_types[idx] = btype
                    if btype == "tool_use":
                        tool_index[idx] = len(tool_index)
                        yield ToolCallStart(tool_index[idx], block.get("id", f"toolu_{idx}"), block.get("name", ""))
                    elif btype in ("thinking", "redacted_thinking"):
                        thinking_blocks[idx] = dict(block)
                    elif btype == "text" and block.get("text"):
                        yield TextDelta(block["text"])
                elif etype == "content_block_delta":
                    idx = int(payload.get("index", 0))
                    delta = payload.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        yield TextDelta(delta.get("text", ""))
                    elif dtype == "input_json_delta":
                        yield ToolCallDelta(tool_index.get(idx, 0), delta.get("partial_json", ""))
                    elif dtype == "thinking_delta":
                        thinking_blocks.setdefault(idx, {"type": "thinking", "thinking": ""})
                        thinking_blocks[idx]["thinking"] = thinking_blocks[idx].get("thinking", "") + delta.get("thinking", "")
                        yield ReasoningDelta(len(delta.get("thinking", "")))
                    elif dtype == "signature_delta":
                        thinking_blocks.setdefault(idx, {"type": "thinking", "thinking": ""})
                        thinking_blocks[idx]["signature"] = thinking_blocks[idx].get("signature", "") + delta.get("signature", "")
                elif etype == "message_delta":
                    u = payload.get("usage") or {}
                    usage.output_tokens = int(u.get("output_tokens") or usage.output_tokens)
                    stop = (payload.get("delta") or {}).get("stop_reason")
                    if thinking_blocks:
                        yield ProviderStateUpdate({"provider": self.id, "thinking": [thinking_blocks[i] for i in sorted(thinking_blocks)]})
                    yield UsageUpdate(usage)
                    if stop:
                        yield Finish(_STOP.get(stop, "other"), stop)
                elif etype == "message_stop":
                    break

    async def list_models(self) -> list[DiscoveredModel]:
        data = await self._get_json(f"{self.base_url}/v1/models?limit=1000", self._headers())
        return [DiscoveredModel(id=m["id"], display_name=m.get("display_name"), raw=m)
                for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
