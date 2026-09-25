"""OpenAI Chat Completions compatible provider.

Covers OpenAI, OpenRouter, Groq, DeepSeek, Together, Mistral, xAI, Google's OpenAI-compatible
endpoint, gateways (LiteLLM, Portkey, Cloudflare) and local servers (Ollama, LM Studio, vLLM,
llama.cpp). Provider identity (base_url + credential) is independent of the model id.
"""

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
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    Usage,
    UsageUpdate,
)

_FINISH = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
}


def _is_host(base_url: str, *hosts: str) -> bool:
    return any(h in base_url for h in hosts)


class OpenAICompatibleProvider(Provider):
    kind = "openai_compatible"

    @property
    def base_url(self) -> str:
        return (self.config.base_url or "").rstrip("/")

    @property
    def requires_credential(self) -> bool:  # type: ignore[override]
        return self.config.api_key is not None

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            **self.config.headers,
            **self.secret_headers,
        }
        if self.credential:
            headers["authorization"] = f"Bearer {self.credential}"
        return headers

    @staticmethod
    def convert_messages(messages: list[ChatMessage], *, vision: bool = True) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_images: list[dict[str, Any]] = []

        def flush() -> None:
            if pending_images:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Images returned by the previous tool calls:"},
                            *pending_images,
                        ],
                    }
                )
                pending_images.clear()

        for m in messages:
            if m.role != "tool":
                flush()
            if m.role == "system":
                out.append({"role": "system", "content": m.content})
            elif m.role == "user":
                if m.images and vision:
                    parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
                    parts += [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{i.media_type};base64,{i.data_b64}"},
                        }
                        for i in m.images
                    ]
                    out.append({"role": "user", "content": parts})
                else:
                    out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.raw_arguments or json.dumps(tc.arguments or {}),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                out.append(item)
            elif m.role == "tool":
                out.append(
                    {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or "(no output)"}
                )
                if m.images and vision:
                    pending_images += [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{i.media_type};base64,{i.data_b64}"},
                        }
                        for i in m.images
                    ]
        flush()
        return out

    def build_body(self, req: ChatRequest) -> dict[str, Any]:
        vision = bool(req.metadata.get("vision", True))
        body: dict[str, Any] = {
            "model": req.model,
            "messages": self.convert_messages(req.messages, vision=vision),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in req.tools
            ]
            if req.tool_choice:
                body["tool_choice"] = req.tool_choice
        if req.max_output_tokens:
            key = "max_completion_tokens" if _is_host(self.base_url, "api.openai.com") else "max_tokens"
            body[key] = req.max_output_tokens
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.reasoning_effort:
            if _is_host(self.base_url, "openrouter.ai"):
                body["reasoning"] = {"effort": req.reasoning_effort, "exclude": True}
            else:
                body["reasoning_effort"] = req.reasoning_effort
        if _is_host(self.base_url, "openrouter.ai"):
            body["usage"] = {"include": True}
        if req.response_schema and not req.tools:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": req.response_schema},
            }
        body.update(self.config.extra_body)
        body.update(req.extra)
        return {k: v for k, v in body.items() if v is not None}

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        self._check_credential()
        body = self.build_body(req)
        started_tools: set[int] = set()
        saw_finish = False
        async with self._post_stream(
            f"{self.base_url}/chat/completions", body, self._headers(), req.model
        ) as response:
            async for sse in iter_sse(response.aiter_lines()):
                data = sse.data.strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ProviderError(
                        f"{self.id}: malformed stream chunk",
                        error_class=ProviderErrorClass.MALFORMED_RESPONSE,
                        provider_id=self.id,
                        model_id=req.model,
                    ) from exc
                if not isinstance(chunk, dict):
                    continue
                if "error" in chunk:
                    raise classify_stream_error(chunk, provider_id=self.id, model_id=req.model)
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    details = usage.get("prompt_tokens_details") or {}
                    cdetails = usage.get("completion_tokens_details") or {}
                    yield UsageUpdate(
                        Usage(
                            input_tokens=int(usage.get("prompt_tokens") or 0),
                            output_tokens=int(usage.get("completion_tokens") or 0),
                            cached_tokens=int(details.get("cached_tokens") or 0),
                            reasoning_tokens=int(cdetails.get("reasoning_tokens") or 0),
                            provider_cost_usd=float(usage["cost"])
                            if isinstance(usage.get("cost"), (int, float))
                            else None,
                        )
                    )
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield TextDelta(delta["content"])
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                    if reasoning:
                        yield ReasoningDelta(len(reasoning))
                    for tc in delta.get("tool_calls") or []:
                        index = int(tc.get("index", 0))
                        fn = tc.get("function") or {}
                        if index not in started_tools:
                            started_tools.add(index)
                            yield ToolCallStart(index, tc.get("id") or f"call_{index}", fn.get("name") or "")
                        if fn.get("arguments"):
                            yield ToolCallDelta(index, fn["arguments"])
                    if choice.get("finish_reason"):
                        saw_finish = True
                        raw = choice["finish_reason"]
                        yield Finish(_FINISH.get(raw, "other"), raw)
        if not saw_finish:
            yield Finish("tool_calls" if started_tools else "stop", None)

    async def list_models(self) -> list[DiscoveredModel]:
        headers = {k: v for k, v in self._headers().items() if k != "accept"}
        data = await self._get_json(f"{self.base_url}/models", headers)
        items = (
            data.get("data", data if isinstance(data, list) else []) if isinstance(data, (dict, list)) else []
        )
        models: list[DiscoveredModel] = []
        for item in items:
            if not isinstance(item, dict) or "id" not in item:
                continue
            pricing = item.get("pricing") or {}
            params = item.get("supported_parameters") or []
            modalities = ((item.get("architecture") or {}).get("input_modalities")) or []
            top = item.get("top_provider") or {}

            def per_mtok(value: Any) -> float | None:
                try:
                    return float(value) * 1_000_000
                except (TypeError, ValueError):
                    return None

            models.append(
                DiscoveredModel(
                    id=str(item["id"]),
                    display_name=item.get("name"),
                    context_window=item.get("context_length") or item.get("context_window"),
                    max_output_tokens=top.get("max_completion_tokens"),
                    supports_tools=("tools" in params) if params else None,
                    supports_vision=("image" in modalities) if modalities else None,
                    input_cost_per_mtok=per_mtok(pricing.get("prompt")),
                    output_cost_per_mtok=per_mtok(pricing.get("completion")),
                    raw=item,
                )
            )
        return models
