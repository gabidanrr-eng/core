"""The single execution path for every tool call (native and MCP)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, cast, get_args

from pydantic import ValidationError

from coremain.domain.states import TaskStatus
from coremain.errors import (
    CoreError,
    LeaseLostError,
    OperationCancelled,
    PolicyDeniedError,
    ToolError,
)
from coremain.providers.types import ToolCall
from coremain.runtime.approvals import Scope
from coremain.security.policy import PolicyDecision
from coremain.tools.base import Tool, ToolContext, ToolResult
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps
from coremain.util.text import one_line

log = logging.getLogger(__name__)
MAX_MODEL_OUTPUT_CHARS = 24_000


class SuspendRequested(Exception):
    """Raised to stop the agent at a durable checkpoint until a human responds."""

    def __init__(
        self,
        kind: str,
        *,
        approval_id: str | None = None,
        question: str | None = None,
        tool_call: ToolCall | None = None,
        reason: str = "",
    ):
        super().__init__(reason or kind)
        self.kind = kind
        self.approval_id = approval_id
        self.question = question
        self.tool_call = tool_call
        self.reason = reason


@dataclass
class ExecutionRecord:
    tool_call_id: str
    tool: str
    status: str
    duration_ms: int
    decision: dict[str, Any] | None


class ToolExecutor:
    def __init__(self, tools: dict[str, Tool]):
        self.tools = tools

    def names(self) -> list[str]:
        return sorted(self.tools)

    async def execute(
        self, call: ToolCall, ctx: ToolContext, *, pre_approved: str | None = None
    ) -> ToolResult:
        svc = ctx.services
        started = time.monotonic()
        tool = self.tools.get(call.name)
        record_id = new_id("tc")
        safe_args = svc.redactor.redact_obj(
            call.arguments if isinstance(call.arguments, dict) else {"raw": call.raw_arguments[:2000]}
        )
        svc.db.execute(
            "INSERT INTO tool_calls(id, project_id, task_id, attempt_id, model_call_id, tool, capability, side_effect, args_json, status, started_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                record_id,
                ctx.project_id,
                ctx.task_id,
                ctx.attempt_id,
                ctx.model_call_id,
                call.name,
                getattr(tool, "capability", None),
                getattr(tool, "side_effect", None),
                dumps(safe_args),
                "running",
                svc.events.clock.now(),
            ),
        )
        decision: PolicyDecision | None = None
        try:
            result, decision = await self._execute(tool, call, ctx, record_id, pre_approved)
        except OperationCancelled:
            self._finish(
                ctx,
                record_id,
                call,
                "cancelled",
                ToolResult.error("cancelled", "cancelled", "cancelled"),
                started,
                decision,
            )
            raise
        except LeaseLostError:
            self._finish(
                ctx,
                record_id,
                call,
                "interrupted",
                ToolResult.error("lease lost", "lease_lost", "interrupted"),
                started,
                decision,
            )
            raise
        except SuspendRequested:
            self._finish(
                ctx,
                record_id,
                call,
                "interrupted",
                ToolResult.error("suspended for human input", "suspended", "interrupted"),
                started,
                decision,
            )
            raise
        result = self._finalize_content(ctx, call, result)
        self._finish(ctx, record_id, call, result.status, result, started, decision)
        return result

    async def _execute(
        self, tool: Tool | None, call: ToolCall, ctx: ToolContext, record_id: str, pre_approved: str | None
    ) -> tuple[ToolResult, PolicyDecision | None]:
        svc = ctx.services
        if tool is None:
            return ToolResult.error(
                f"unknown tool '{call.name}'. Available tools: {', '.join(self.names())}", "unknown_tool"
            ), None
        if call.parse_error:
            return ToolResult.error(
                f"arguments for {call.name} were not valid JSON ({call.parse_error}); resend valid JSON arguments",
                "invalid_arguments",
            ), None
        try:
            args = tool.Input.model_validate(call.arguments or {})
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors()[:6]
            )
            return ToolResult.error(
                f"invalid arguments for {call.name}: {problems}", "invalid_arguments"
            ), None
        request = tool.policy_request(ctx, args)
        decision: PolicyDecision | None = None
        if request is not None:
            decision = svc.policy.evaluate(request)
            if decision.decision == "deny":
                svc.events.emit(
                    "tool.denied",
                    project_id=ctx.project_id,
                    session_id=ctx.session_id,
                    task_id=ctx.task_id,
                    attempt_id=ctx.attempt_id,
                    level="warning",
                    data={
                        "tool": call.name,
                        "target": request.target[:300],
                        "reason": decision.reason,
                        "source": decision.source,
                    },
                )
                return ToolResult.error(
                    f"denied by policy: {decision.reason}", "policy_denied", "denied"
                ), decision
            if decision.decision == "ask":
                approved, why = await self._approve(
                    tool, args, request.target, decision, ctx, call, record_id, pre_approved
                )
                if not approved:
                    return ToolResult.error(f"not approved: {why}", "approval_rejected", "denied"), decision
        svc.events.ephemeral(
            "tool.started",
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            attempt_id=ctx.attempt_id,
            data={
                "tool": call.name,
                "summary": one_line(tool.summarize(args), 200),
                "tool_call_id": record_id,
            },
        )
        timeout = tool.effective_timeout(args)
        try:
            result = await asyncio.wait_for(ctx.cancel.run(tool.run(ctx, args)), timeout=timeout)
        except TimeoutError:
            return ToolResult.error(
                f"{call.name} timed out after {timeout:.0f}s", "timeout", "timeout"
            ), decision
        except (OperationCancelled, LeaseLostError, SuspendRequested):
            raise
        except PolicyDeniedError as exc:
            return ToolResult.error(f"denied: {exc.message}", exc.code, "denied"), decision
        except ToolError as exc:
            return ToolResult.error(
                exc.message + (f" (hint: {exc.hint})" if exc.hint else ""), exc.error_class
            ), decision
        except CoreError as exc:
            return ToolResult.error(exc.message, exc.code), decision
        except Exception as exc:
            log.exception("tool %s crashed", call.name)
            svc.events.emit(
                "tool.crashed",
                project_id=ctx.project_id,
                task_id=ctx.task_id,
                attempt_id=ctx.attempt_id,
                level="error",
                data={"tool": call.name, "error": f"{type(exc).__name__}: {exc}"},
            )
            return ToolResult.error(
                f"internal error in {call.name}: {type(exc).__name__}: {exc}", "internal_error"
            ), decision
        return result, decision

    async def _approve(
        self,
        tool: Tool,
        args: Any,
        target: str,
        decision: PolicyDecision,
        ctx: ToolContext,
        call: ToolCall,
        record_id: str,
        pre_approved: str | None,
    ) -> tuple[bool, str]:
        svc = ctx.services
        if pre_approved:
            approval = svc.approvals.get(pre_approved)
            if approval.status == "approved":
                return True, "approved"
            if approval.status in {"rejected", "cancelled", "expired"}:
                return False, approval.reason or approval.status
        approval = svc.approvals.request(
            capability=tool.capability,
            summary=one_line(tool.summarize(args), 300),
            request={
                "tool": call.name,
                "target": target,
                "reason": decision.reason,
                "risk": decision.risk,
                "arguments": svc.redactor.redact_obj(call.arguments or {}),
            },
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            attempt_id=ctx.attempt_id,
            tool_call_id=record_id,
        )
        if ctx.task_id:
            svc.tasks.transition(
                ctx.task_id,
                TaskStatus.AWAITING_APPROVAL,
                reason=f"approval needed: {approval.summary}",
                fence=ctx.fence,
            )
        handler = svc.approval_handler
        if handler is None:
            if svc.config.permissions.non_interactive_ask == "suspend" and ctx.task_id:
                raise SuspendRequested(
                    "approval", approval_id=approval.id, tool_call=call, reason=approval.summary
                )
            svc.approvals.decide(
                approval.id, False, actor="runtime", reason="approval required but running non-interactively"
            )
            if ctx.task_id:
                svc.tasks.transition(
                    ctx.task_id,
                    TaskStatus.RUNNING,
                    reason="approval auto-rejected (non-interactive)",
                    fence=ctx.fence,
                )
            return False, "approval required but running non-interactively"
        interactive = asyncio.ensure_future(handler(approval))
        external = asyncio.ensure_future(svc.approvals.wait(approval.id, ctx.cancel))
        try:
            done, _ = await asyncio.wait({interactive, external}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for fut in (interactive, external):
                if not fut.done():
                    fut.cancel()
        if interactive in done and not interactive.cancelled() and interactive.exception() is None:
            answer = interactive.result()
            scope = cast(Scope, answer.scope if answer.scope in get_args(Scope) else "once")
            final = svc.approvals.decide(approval.id, answer.approved, scope=scope, reason=answer.reason)
        elif external in done and external.exception() is None:
            final = external.result()
        else:
            exc = (
                interactive.exception() if interactive.done() and not interactive.cancelled() else None
            ) or external.exception()
            if isinstance(exc, OperationCancelled):
                raise exc
            final = svc.approvals.decide(
                approval.id, False, actor="runtime", reason=f"approval handler failed: {exc}"
            )
        if ctx.task_id:
            svc.tasks.transition(
                ctx.task_id, TaskStatus.RUNNING, reason=f"approval {final.status}", fence=ctx.fence
            )
        return final.status == "approved", final.reason or final.status

    def _finalize_content(self, ctx: ToolContext, call: ToolCall, result: ToolResult) -> ToolResult:
        svc = ctx.services
        content = svc.redactor.redact(result.content or "")
        if len(content) > MAX_MODEL_OUTPUT_CHARS:
            if result.artifact_id is None:
                art = svc.artifacts.put_text(
                    content,
                    kind="tool_output",
                    name=call.name,
                    project_id=ctx.project_id,
                    task_id=ctx.task_id,
                    attempt_id=ctx.attempt_id,
                    redact=False,
                )
                result.artifact_id = art.id
            head = MAX_MODEL_OUTPUT_CHARS * 2 // 3
            tail = MAX_MODEL_OUTPUT_CHARS - head
            content = (
                content[:head]
                + f"\n… [{len(content) - MAX_MODEL_OUTPUT_CHARS} chars omitted; full output in artifact "
                f"{result.artifact_id}] …\n" + content[-tail:]
            )
        result.content = content
        return result

    def _finish(
        self,
        ctx: ToolContext,
        record_id: str,
        call: ToolCall,
        status: str,
        result: ToolResult,
        started: float,
        decision: PolicyDecision | None,
    ) -> None:
        svc = ctx.services
        duration_ms = int((time.monotonic() - started) * 1000)
        summary = one_line(result.content or "", 400)
        svc.db.execute(
            "UPDATE tool_calls SET status = ?, error_class = ?, error_message = ?, decision_json = ?, output_summary = ?, artifact_id = ?, "
            "ended_at = ?, duration_ms = ? WHERE id = ?",
            (
                status,
                result.error_class,
                None if result.ok else summary,
                dumps(decision.to_dict() if decision else {}),
                summary,
                result.artifact_id,
                svc.events.clock.now(),
                duration_ms,
                record_id,
            ),
        )
        svc.events.emit(
            "tool.call",
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            attempt_id=ctx.attempt_id,
            level="info" if result.ok else "warning",
            data={
                "tool_call_id": record_id,
                "tool": call.name,
                "status": status,
                "ok": result.ok,
                "error_class": result.error_class,
                "duration_ms": duration_ms,
                "summary": summary[:300],
                "node": ctx.node,
                "args": svc.redactor.redact_obj(
                    {
                        k: (v if not isinstance(v, str) else one_line(v, 200))
                        for k, v in (call.arguments or {}).items()
                    }
                )
                if isinstance(call.arguments, dict)
                else {},
            },
        )
