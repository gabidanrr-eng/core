"""The model ↔ tool loop.

One loop run serves one workflow node with one routed model. It streams the model, executes
tool calls through ``ToolExecutor`` (policy, approvals, audit), and ends when the model calls
the node's structured output tool. Safeguards: turn and task budgets, bounded repair of
invalid structured output, detection of repeated identical calls, context compaction near
the model's window, cancellation at every await, and durable conversation checkpoints so a
suspended or interrupted node can resume.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from coremain.events import EventLog
from coremain.providers.client import CallContext, ModelClient, compute_cost
from coremain.providers.errors import ProviderError, ProviderErrorClass
from coremain.providers.types import ChatMessage, ChatRequest, ToolCall, ToolSpec, Usage
from coremain.routing.router import RouteDecision
from coremain.runtime.budgets import BudgetTracker
from coremain.tools.base import Tool, ToolContext
from coremain.tools.executor import SuspendRequested, ToolExecutor
from coremain.util.text import estimate_tokens, one_line, truncate_middle

INTERRUPTED_RESULT = (
    "[interrupted: the runtime stopped before this tool call completed; its effects are unknown. "
    "Inspect the workspace state before continuing.]"
)


@dataclass
class AgentSpec:
    role: str
    node: str
    system: str
    context_text: str
    tools: list[Tool]
    route: RouteDecision
    output_tool: str | None
    max_turns: int
    max_structured_repairs: int = 2
    compaction_threshold: float = 0.72
    messages: list[ChatMessage] | None = None
    pending_calls: list[ToolCall] = field(default_factory=list)
    pending_approval: str | None = None


@dataclass
class AgentOutcome:
    status: str  # completed | no_output | budget_exhausted | suspended
    payload: dict[str, Any] | None
    text: str
    messages: list[ChatMessage]
    turns: int
    usage: Usage
    model_ref: str
    tool_calls: int = 0
    malformed: int = 0
    suspend: SuspendRequested | None = None
    pending_calls: list[ToolCall] = field(default_factory=list)
    note: str | None = None


class _Suspended(Exception):
    def __init__(self, signal: SuspendRequested, remaining: list[ToolCall]):
        super().__init__(signal.kind)
        self.signal = signal
        self.remaining = remaining


def normalize_conversation(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Drop provider-private state and close tool calls that never got a result."""
    out: list[ChatMessage] = []
    answered = {m.tool_call_id for m in messages if m.role == "tool"}
    for m in messages:
        m.provider_state = None
        out.append(m)
        if m.role == "assistant":
            for call in m.tool_calls:
                if call.id not in answered:
                    out.append(
                        ChatMessage(
                            "tool", INTERRUPTED_RESULT, tool_call_id=call.id, name=call.name, is_error=True
                        )
                    )
                    answered.add(call.id)
    return out


def conversation_tokens(messages: list[ChatMessage]) -> int:
    return sum(
        estimate_tokens(m.content)
        + 20 * len(m.tool_calls)
        + sum(len(t.raw_arguments) // 4 for t in m.tool_calls)
        for m in messages
    )


def compact(messages: list[ChatMessage], limit_tokens: int) -> tuple[list[ChatMessage], int]:
    """Elide old tool outputs (keeping structure), then truncate the compiled context if needed."""
    if conversation_tokens(messages) <= limit_tokens:
        return messages, 0
    elided = 0
    keep_tail = 8
    middle = messages[2:-keep_tail] if len(messages) > keep_tail + 2 else []
    for m in middle:
        if m.role == "tool" and len(m.content) > 400:
            m.content = f"[elided earlier output of {m.name or 'tool'} ({len(m.content)} chars); re-run the tool if you need it]"
            m.images = []
            elided += 1
        elif m.role == "assistant" and len(m.content) > 1200:
            m.content = truncate_middle(m.content, 1200)
            elided += 1
    if conversation_tokens(messages) > limit_tokens and len(messages) > 1 and messages[1].role == "user":
        over = conversation_tokens(messages) - limit_tokens
        target = max(4000, len(messages[1].content) - over * 4)
        messages[1].content = truncate_middle(messages[1].content, target)
        elided += 1
    return messages, elided


class AgentLoop:
    def __init__(
        self,
        client: ModelClient,
        events: EventLog,
        budget: BudgetTracker,
        checkpoint: Callable[[list[ChatMessage], list[ToolCall]], None] | None = None,
    ):
        self.client = client
        self.events = events
        self.budget = budget
        self.checkpoint = checkpoint

    async def run(self, spec: AgentSpec, ctx: ToolContext, call_ctx: CallContext) -> AgentOutcome:
        model = spec.route.model
        executor = ToolExecutor({t.name: t for t in spec.tools})
        tool_specs = [ToolSpec(t.name, t.description, t.schema()) for t in spec.tools]
        if spec.messages:
            messages = normalize_conversation(list(spec.messages))
        else:
            messages = [ChatMessage("system", spec.system), ChatMessage("user", spec.context_text)]
        usage = Usage()
        state: dict[str, Any] = {
            "turns": 0,
            "tools": 0,
            "malformed": 0,
            "invalid": 0,
            "payload": None,
            "text": "",
        }
        nudged = False
        recent: deque[tuple[str, str]] = deque(maxlen=8)
        window = model.config.context_window
        ids: dict[str, Any] = {
            "project_id": ctx.project_id,
            "session_id": ctx.session_id,
            "task_id": ctx.task_id,
            "attempt_id": ctx.attempt_id,
        }

        def outcome(status: str, note: str | None = None, **kw: Any) -> AgentOutcome:
            return AgentOutcome(
                status,
                state["payload"],
                state["text"],
                messages,
                state["turns"],
                usage,
                model.ref,
                state["tools"],
                state["malformed"],
                note=note,
                **kw,
            )

        async def run_calls(calls: list[ToolCall], first_pre_approved: str | None) -> bool:
            for i, call in enumerate(calls):
                signature = (call.name, call.raw_arguments.strip())
                repeats = sum(1 for s in recent if s == signature)
                recent.append(signature)
                if repeats >= 2 and not call.name.startswith("submit_"):
                    messages.append(
                        ChatMessage(
                            "tool",
                            "repeated identical call detected (3rd time); the result will not change. "
                            "Change your approach or finish.",
                            tool_call_id=call.id,
                            name=call.name,
                            is_error=True,
                        )
                    )
                    self.events.emit(
                        "agent.loop_detected",
                        level="warning",
                        data={"tool": call.name, "node": spec.node},
                        **ids,
                    )
                    continue
                try:
                    result = await executor.execute(
                        call, ctx, pre_approved=first_pre_approved if i == 0 else None
                    )
                except SuspendRequested as sus:
                    sus.tool_call = call
                    remaining = calls[i:]
                    if self.checkpoint is not None:
                        self.checkpoint(messages, remaining)
                    raise _Suspended(sus, remaining) from None
                state["tools"] += 1
                messages.append(
                    ChatMessage(
                        "tool", result.content, tool_call_id=call.id, name=call.name, is_error=not result.ok
                    )
                )
                if call.name == spec.output_tool:
                    if result.ok and result.terminal:
                        state["payload"] = result.payload
                    else:
                        state["invalid"] += 1
            return state["payload"] is not None

        try:
            if spec.pending_calls and await run_calls(list(spec.pending_calls), spec.pending_approval):
                return outcome("completed")
            while True:
                ctx.cancel.raise_if_cancelled()
                if state["turns"] >= spec.max_turns:
                    return outcome("budget_exhausted", f"node turn limit ({spec.max_turns}) reached")
                self.budget.check()
                messages, elided = compact(messages, int(window * spec.compaction_threshold))
                if elided:
                    self.events.emit("context.compacted", data={"node": spec.node, "elided": elided}, **ids)
                request = ChatRequest(
                    model=model.model_id,
                    messages=messages,
                    tools=tool_specs,
                    tool_choice="auto" if tool_specs else None,
                )

                def on_delta(text: str, _events: EventLog = self.events) -> None:
                    _events.ephemeral(
                        "model.delta", data={"text": text, "node": spec.node, "role": spec.role}, **ids
                    )

                try:
                    response = await self.client.complete(
                        model, request, call_ctx, cancel=ctx.cancel, on_delta=on_delta
                    )
                except ProviderError as exc:
                    if exc.error_class != ProviderErrorClass.CONTEXT_OVERFLOW or len(messages) <= 3:
                        raise
                    messages, _ = compact(messages, int(window * 0.45))
                    self.events.emit(
                        "context.overflow_recovery", level="warning", data={"node": spec.node}, **ids
                    )
                    request.messages = messages
                    response = await self.client.complete(
                        model, request, call_ctx, cancel=ctx.cancel, on_delta=on_delta
                    )
                state["turns"] += 1
                usage.add(response.usage)
                cost, _ = compute_cost(model, response.usage)
                self.budget.record(model.ref, response.usage.input_tokens, response.usage.output_tokens, cost)
                ctx.model_call_id = response.model_call_id
                state["malformed"] += sum(1 for c in response.tool_calls if c.parse_error)
                messages.append(
                    ChatMessage(
                        "assistant",
                        response.text,
                        list(response.tool_calls),
                        provider_state=response.provider_state,
                    )
                )
                if response.text.strip():
                    state["text"] = response.text
                    self.events.emit(
                        "agent.message",
                        data={
                            "node": spec.node,
                            "role": spec.role,
                            "model": model.ref,
                            "text": one_line(response.text, 600),
                        },
                        **ids,
                    )
                if not response.tool_calls:
                    if spec.output_tool and not nudged:
                        nudged = True
                        messages.append(
                            ChatMessage(
                                "user",
                                "Continue working with the tools. When the work is complete, you must call "
                                f"`{spec.output_tool}` with the structured result.",
                            )
                        )
                        continue
                    if spec.output_tool:
                        return outcome("no_output", "model stopped without calling the output tool")
                    return outcome("completed")
                done = await run_calls(list(response.tool_calls), None)
                if self.checkpoint is not None:
                    self.checkpoint(messages, [])
                if done:
                    return outcome("completed")
                if state["invalid"] > spec.max_structured_repairs:
                    return outcome(
                        "no_output", f"structured output invalid after {state['invalid']} attempts"
                    )
        except _Suspended as s:
            return outcome("suspended", suspend=s.signal, pending_calls=s.remaining)
