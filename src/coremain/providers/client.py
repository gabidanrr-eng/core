"""ModelClient: one accounted, cancellable, bounded model call.

* Records every call in ``model_calls`` (provider, model, role, usage, cost, latency, TTFT,
  retries, error class) and emits durable events.
* Retries only transient classes, within ``max_retries`` of the provider and honouring
  ``Retry-After``; never retries auth, billing, invalid requests or unavailable models.
* Fallback happens only through rules the user configured (``routing.fallback``) and is
  announced with a ``route.fallback`` event.
* Streaming text deltas are published as ephemeral events for live clients.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from coremain.config.schema import CoreConfig
from coremain.errors import OperationCancelled
from coremain.events import EventLog
from coremain.providers.base import parse_json_arguments
from coremain.providers.errors import ProviderError, ProviderErrorClass
from coremain.providers.registry import ModelProfile, ProviderRegistry
from coremain.providers.types import (
    ChatRequest,
    ChatResponse,
    Finish,
    ProviderStateUpdate,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    Usage,
    UsageUpdate,
)
from coremain.runtime.cancel import CancelToken
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id


@dataclass
class CallContext:
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    node: str | None = None
    role: str | None = None
    purpose: str | None = None
    context_snapshot_id: str | None = None
    prompt_version: str | None = None


DeltaCallback = Callable[[str], None]


def compute_cost(model: ModelProfile, usage: Usage) -> tuple[float | None, str | None]:
    if usage.provider_cost_usd is not None:
        return usage.provider_cost_usd, "provider"
    cost = model.config.cost
    if cost.input_per_mtok is None and cost.output_per_mtok is None:
        return None, None
    uncached = max(0, usage.input_tokens - usage.cached_tokens)
    total = uncached * (cost.input_per_mtok or 0) / 1e6 + usage.output_tokens * (cost.output_per_mtok or 0) / 1e6
    total += usage.cached_tokens * (cost.cached_input_per_mtok if cost.cached_input_per_mtok is not None else cost.input_per_mtok or 0) / 1e6
    return round(total, 6), "declared"


class ModelClient:
    def __init__(self, registry: ProviderRegistry, db: Database, events: EventLog, clock: Clock, config: CoreConfig):
        self.registry = registry
        self.db = db
        self.events = events
        self.clock = clock
        self.config = config
        self.on_failure: Callable[[ProviderError, ModelProfile, CallContext], None] | None = None

    def fallback_chain(self, model: ModelProfile, error: ProviderError, tried: set[str]) -> ModelProfile | None:
        for rule in self.config.routing.fallback:
            if error.error_class in rule.on:
                for key in rule.to:
                    if key in tried:
                        continue
                    try:
                        candidate = self.registry.model(key)
                    except Exception:  # noqa: BLE001 - misconfigured fallback entries are skipped visibly below
                        continue
                    if candidate.config.enabled:
                        return candidate
        return None

    async def complete(self, model: ModelProfile, req: ChatRequest, ctx: CallContext, *, cancel: CancelToken,
                       on_delta: DeltaCallback | None = None) -> ChatResponse:
        tried: set[str] = set()
        current = model
        while True:
            tried.add(current.key)
            try:
                return await self._complete_one(current, req, ctx, cancel=cancel, on_delta=on_delta)
            except ProviderError as exc:
                nxt = self.fallback_chain(current, exc, tried)
                if nxt is None:
                    raise
                self.events.emit("route.fallback", project_id=ctx.project_id, session_id=ctx.session_id, task_id=ctx.task_id,
                                 attempt_id=ctx.attempt_id, level="warning",
                                 data={"from": current.ref, "to": nxt.ref, "error_class": exc.error_class.value,
                                       "reason": "configured routing.fallback rule"})
                req = ChatRequest(**{**req.__dict__, "model": nxt.model_id})
                current = nxt

    async def _complete_one(self, model: ModelProfile, req: ChatRequest, ctx: CallContext, *, cancel: CancelToken,
                            on_delta: DeltaCallback | None) -> ChatResponse:
        provider = self.registry.provider(model.provider_id)
        max_retries = self.config.providers[model.provider_id].max_retries
        call_id = new_id("mc")
        started_wall = self.clock.now()
        self.db.execute(
            "INSERT INTO model_calls(id, project_id, task_id, attempt_id, node, role, purpose, provider_id, model_key, model_id, status, "
            "context_snapshot_id, prompt_version, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (call_id, ctx.project_id, ctx.task_id, ctx.attempt_id, ctx.node, ctx.role, ctx.purpose, model.provider_id, model.key,
             model.model_id, "running", ctx.context_snapshot_id, ctx.prompt_version, started_wall),
        )
        self.events.ephemeral("model.started", project_id=ctx.project_id, session_id=ctx.session_id, task_id=ctx.task_id,
                              attempt_id=ctx.attempt_id, data={"model": model.ref, "role": ctx.role, "node": ctx.node,
                                                               "model_call_id": call_id, "kind": provider.kind})
        req.model = model.model_id
        req.metadata.setdefault("role", ctx.role)
        req.metadata.setdefault("node", ctx.node)
        req.metadata["vision"] = model.config.capabilities.vision
        req.metadata["reasoning"] = model.config.capabilities.reasoning
        req.metadata["prompt_caching"] = model.config.capabilities.prompt_caching
        if req.reasoning_effort is None:
            req.reasoning_effort = model.config.capabilities.reasoning_effort
        if req.max_output_tokens is None:
            req.max_output_tokens = min(model.config.max_output_tokens, 16_384)
        if req.temperature is None and "temperature" in model.config.params:
            req.temperature = float(model.config.params["temperature"])
        retries = 0
        while True:
            t0 = time.monotonic()
            try:
                response = await cancel.run(self._stream_once(provider, req, t0, ctx, on_delta))
                response.retries = retries
                response.model_ref = model.ref
                response.model_call_id = call_id
                self._finish(call_id, model, response, None)
                return response
            except OperationCancelled:
                self.db.execute("UPDATE model_calls SET status = 'cancelled', ended_at = ?, retries = ? WHERE id = ?",
                                (self.clock.now(), retries, call_id))
                raise
            except ProviderError as exc:
                exc.provider_id = exc.provider_id or model.provider_id
                exc.model_id = exc.model_id or model.model_id
                if exc.transient and retries < max_retries:
                    retries += 1
                    delay = exc.retry_after if exc.retry_after is not None else min(30.0, (2 ** retries) * 0.5 + random.uniform(0, 0.5))
                    self.events.emit("model.retry", project_id=ctx.project_id, task_id=ctx.task_id, attempt_id=ctx.attempt_id,
                                     level="warning", data={"model": model.ref, "error_class": exc.error_class.value, "retry": retries,
                                                            "delay_s": round(delay, 2)})
                    await cancel.sleep(min(delay, 60.0))
                    continue
                self.db.execute(
                    "UPDATE model_calls SET status = 'error', error_class = ?, error_message = ?, retries = ?, ended_at = ?, "
                    "latency_ms = ? WHERE id = ?",
                    (exc.error_class.value, exc.message[:500], retries, self.clock.now(), int((time.monotonic() - t0) * 1000), call_id),
                )
                self.events.emit("model.call.failed", project_id=ctx.project_id, session_id=ctx.session_id, task_id=ctx.task_id,
                                 attempt_id=ctx.attempt_id, level="error",
                                 data={"model": model.ref, "error_class": exc.error_class.value, "message": exc.message[:400],
                                       "status_code": exc.status_code, "retries": retries, "hint": exc.hint})
                if self.on_failure is not None:
                    self.on_failure(exc, model, ctx)
                raise

    async def _stream_once(self, provider: Any, req: ChatRequest, t0: float, ctx: CallContext,
                           on_delta: DeltaCallback | None) -> ChatResponse:
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage()
        finish = "stop"
        ttft: int | None = None
        state: dict[str, Any] | None = None
        async for event in provider.stream(req):
            if isinstance(event, TextDelta):
                if ttft is None:
                    ttft = int((time.monotonic() - t0) * 1000)
                text_parts.append(event.text)
                if on_delta is not None:
                    on_delta(event.text)
            elif isinstance(event, ReasoningDelta):
                if ttft is None:
                    ttft = int((time.monotonic() - t0) * 1000)
            elif isinstance(event, ToolCallStart):
                if ttft is None:
                    ttft = int((time.monotonic() - t0) * 1000)
                calls[event.index] = {"id": event.id, "name": event.name, "args": []}
            elif isinstance(event, ToolCallDelta):
                calls.setdefault(event.index, {"id": f"call_{event.index}", "name": "", "args": []})["args"].append(event.arguments)
            elif isinstance(event, UsageUpdate):
                usage = event.usage
            elif isinstance(event, Finish):
                finish = event.reason
            elif isinstance(event, ProviderStateUpdate):
                state = event.state
        tool_calls: list[ToolCall] = []
        for index in sorted(calls):
            raw = "".join(calls[index]["args"])
            args, err = parse_json_arguments(raw)
            if not calls[index]["name"]:
                raise ProviderError("provider streamed a tool call without a name", error_class=ProviderErrorClass.MALFORMED_RESPONSE,
                                    provider_id=provider.id, model_id=req.model)
            tool_calls.append(ToolCall(calls[index]["id"], calls[index]["name"], args, raw, err))
        if finish == "content_filter" and not text_parts and not tool_calls:
            raise ProviderError("provider filtered the response", error_class=ProviderErrorClass.CONTENT_FILTERED,
                                provider_id=provider.id, model_id=req.model)
        return ChatResponse("".join(text_parts), tool_calls, usage, finish, state, int((time.monotonic() - t0) * 1000), ttft)

    def _finish(self, call_id: str, model: ModelProfile, response: ChatResponse, error: ProviderError | None) -> None:
        cost, source = compute_cost(model, response.usage)
        malformed = sum(1 for c in response.tool_calls if c.parse_error)
        self.db.execute(
            "UPDATE model_calls SET status = 'ok', finish_reason = ?, input_tokens = ?, output_tokens = ?, cached_tokens = ?, "
            "reasoning_tokens = ?, cost_usd = ?, cost_source = ?, latency_ms = ?, ttft_ms = ?, retries = ?, tool_call_count = ?, "
            "error_class = ?, ended_at = ? WHERE id = ?",
            (response.finish_reason, response.usage.input_tokens, response.usage.output_tokens, response.usage.cached_tokens,
             response.usage.reasoning_tokens, cost, source, response.latency_ms, response.ttft_ms, response.retries,
             len(response.tool_calls), "malformed_tool_arguments" if malformed else None, self.clock.now(), call_id),
        )
