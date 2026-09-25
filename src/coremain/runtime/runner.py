"""TaskRunner: executes one task attempt through its workflow under a fenced lease.

Every failure class maps to an explicit task state: provider credential/billing/availability
problems → ``blocked``; human decisions → ``awaiting_approval``/``needs_input``; cancellation
→ ``cancelled``; lost lease → the worker stops writing immediately; missing or failing
evidence → ``incomplete``; unexpected crashes during mutating work → ``unknown``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coremain.agent.loop import AgentLoop, AgentOutcome, AgentSpec
from coremain.agent.prompts import PROMPT_VERSION, system_prompt
from coremain.agent.workflow import Node, Workflow, build_workflows
from coremain.context.compiler import CompiledContext, ContextRequest
from coremain.domain.models import Attempt, Project, Task
from coremain.domain.states import AttemptStatus, TaskStatus
from coremain.errors import (
    BudgetExceededError,
    CoreError,
    GateFailedError,
    LeaseLostError,
    OperationCancelled,
    WorkspaceConflictError,
    WorkspaceError,
)
from coremain.providers.client import CallContext
from coremain.providers.errors import ProviderError
from coremain.providers.types import ChatMessage, ToolCall
from coremain.review.engine import verdict_for
from coremain.review.static_checks import Finding, static_review
from coremain.routing.router import RouteDecision, RouteRequirements
from coremain.runtime.budgets import BudgetTracker
from coremain.runtime.cancel import CancelToken
from coremain.runtime.leases import Fence
from coremain.runtime.report import render_report
from coremain.runtime.tasks import task_resource
from coremain.runtime.toolsets import tools_for
from coremain.security.paths import PathViolation, resolve_within
from coremain.tools.base import ToolContext
from coremain.util.text import estimate_tokens, one_line
from coremain.verify.evidence import EvidenceKind, GateContext, Requirement, Trust
from coremain.workspaces.manager import Workspace

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime

log = logging.getLogger(__name__)
LEASE_TTL_S = 60.0
HEARTBEAT_S = 10.0


@dataclass
class NodeResult:
    outcome: str
    note: str | None = None
    output: Any = None


@dataclass
class TaskRun:
    rt: CoreRuntime
    task: Task
    attempt: Attempt
    fence: Fence
    project: Project
    workspace: Workspace
    workflow: Workflow
    cancel: CancelToken
    budget: BudgetTracker
    profile: dict[str, Any]
    state: dict[str, Any]
    requirements: list[Requirement]
    read_hashes: dict[str, str] = field(default_factory=dict)
    files_read: set[str] = field(default_factory=set)
    changed_paths: set[str] = field(default_factory=set)
    commands_run: list[dict[str, Any]] = field(default_factory=list)
    implementer_model: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def classification(self) -> dict[str, Any]:
        return self.task.decision.get("classification", {})

    @property
    def changes_code(self) -> bool:
        return bool(self.classification.get("changes_code", True))

    @property
    def outputs(self) -> dict[str, Any]:
        return self.state.setdefault("outputs", {})

    def ids(self) -> dict[str, Any]:
        return {"project_id": self.project.id, "session_id": self.task.session_id, "task_id": self.task.id, "attempt_id": self.attempt.id}

    def emit(self, event_kind: str, level: str = "info", **data: Any) -> None:
        self.rt.events.emit(event_kind, level=level, data=data, **self.ids())

    def transition(self, status: TaskStatus, reason: str, **kw: Any) -> Task:
        self.task = self.rt.tasks.transition(self.task.id, status, reason=reason, fence=self.fence, **kw)
        return self.task

    def ctx(self, role: str, node: str) -> ToolContext:
        services = self.rt.tool_services(self.project, self.profile)
        services.spawn_subtask = self._spawn if self.task.depth < self.rt.config.budgets.max_subtask_depth else None
        ctx = ToolContext(project_id=self.project.id, session_id=self.task.session_id, task_id=self.task.id,
                          attempt_id=self.attempt.id, workspace=self.workspace, fence=self.fence, cancel=self.cancel.child(), role=role,
                          services=services, node=node, depth=self.task.depth)
        ctx.read_hashes = self.read_hashes
        ctx.files_read = self.files_read
        ctx.changed_paths = self.changed_paths
        ctx.commands_run = self.commands_run
        ctx.extra_env = self.rt.workspace_env(self.workspace, self.profile)
        return ctx

    async def _spawn(self, ctx: ToolContext, title: str, instructions: str, kind: str) -> str:
        children = self.rt.tasks.list(parent_task_id=self.task.id, limit=100)
        if len(children) >= self.rt.config.budgets.max_fanout:
            return f"not spawned: fan-out budget ({self.rt.config.budgets.max_fanout}) reached for this task"
        child = self.rt.tasks.create(project_id=self.project.id, session_id=self.task.session_id, parent_task_id=self.task.id,
                                     title=title[:200], description=instructions, kind="question" if kind != "review" else "review",
                                     options={"workspace": "canonical_readonly", "parent_workspace": str(self.workspace.path)})
        self.emit("task.spawned", child=child.id, title=title)
        final = await self.rt.run_task(child.id)
        answer = final.result_summary or "(no result)"
        return f"subtask {child.id} finished as {final.status.value}: {answer[:6000]}"


class TaskRunner:
    def __init__(self, rt: CoreRuntime, worker_name: str):
        self.rt = rt
        self.worker = worker_name
        self.workflows = build_workflows(rt.config.budgets.max_repair_iterations)
        self.handlers = {
            "plan": self._plan, "propose": self._propose, "synthesize": self._synthesize, "implement": self._implement,
            "verify": self._verify, "review": self._review, "answer": self._answer, "review_existing": self._review_existing,
            "reproduce": self._reproduce, "finalize": self._finalize,
        }

    # ================================================================ lifecycle
    async def execute(self, task_id: str, cancel: CancelToken) -> Task:
        rt = self.rt
        owner = f"{rt.runtime_id}:{self.worker}"
        fence = rt.leases.acquire(task_resource(task_id), owner, LEASE_TTL_S)
        if fence is None:
            return rt.tasks.get(task_id)
        heartbeat: asyncio.Task[None] | None = None
        run: TaskRun | None = None
        try:
            task = rt.tasks.get(task_id)
            if task.status != TaskStatus.QUEUED:
                return task
            project = rt.projects.get(task.project_id)
            profile = await rt.profile_for(project)
            previous = rt.tasks.attempts(task_id)
            resume_from = previous[-1] if previous and previous[-1].status in {AttemptStatus.SUSPENDED, AttemptStatus.INTERRUPTED} else None
            workflow = self.workflows[task.decision.get("workflow", "direct")]
            workspace = await self._workspace_for(task, project)
            if task.workspace_id != workspace.id:
                task = rt.tasks.update_fields(task_id, fence=fence, workspace_id=workspace.id)
            checkpoint = dict(resume_from.checkpoint) if resume_from else {"node": workflow.start, "visits": {}, "outputs": {}, "history": []}
            base_fp = await rt.workspaces.fingerprint(workspace) if workspace.base_ref else None
            task, attempt = rt.tasks.begin_attempt(task_id, fence, runtime_id=rt.runtime_id, workflow=workflow.name,
                                                   workspace_id=workspace.id, resumed_from=resume_from.id if resume_from else None,
                                                   base_fingerprint=base_fp, checkpoint=checkpoint)
            budget = BudgetTracker.resume(rt.config.budgets, rt.clock, [a.usage for a in previous])
            requirements = [Requirement.from_dict(r) for r in task.contract.get("requirements", [])]
            run = TaskRun(rt, task, attempt, fence, project, workspace, workflow, cancel, budget, profile, checkpoint, requirements)
            if resume_from is not None:
                run.emit("task.resumed", from_attempt=resume_from.id, node=checkpoint.get("node"),
                         reason="resuming from durable checkpoint")
            heartbeat = asyncio.create_task(self._heartbeat(run))
            await self._run_workflow(run)
        except LeaseLostError as exc:
            log.warning("lease lost for %s: %s", task_id, exc)
            cancel.cancel("lease lost")
        except Exception as exc:
            if run is None:
                log.exception("task %s failed before its attempt started", task_id)
                with contextlib.suppress(CoreError):
                    rt.tasks.transition(task_id, TaskStatus.FAILED, reason=f"could not start: {exc}", fence=fence)
            else:
                await self._handle_error(run, exc)
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await heartbeat
            rt.leases.release(fence)
        return rt.tasks.get(task_id)

    async def _heartbeat(self, run: TaskRun) -> None:
        rt = self.rt
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            if not rt.tasks.heartbeat(run.attempt.id, run.fence, LEASE_TTL_S):
                run.cancel.cancel("lease lost")
                return
            task = rt.tasks.get(run.task.id)
            if task.cancel_requested_at is not None and not run.cancel.cancelled:
                run.cancel.cancel(task.cancel_reason or "cancelled")

    async def _workspace_for(self, task: Task, project: Project) -> Workspace:
        rt = self.rt
        if task.workspace_id:
            try:
                ws = rt.workspaces.get(task.workspace_id)
                if ws.path.exists() and ws.status in {"active", "preserved"}:
                    if ws.status == "preserved":
                        ws = rt.workspaces.set_status(ws, "active", reactivated=True)
                    return ws
            except CoreError:
                pass
        changes = bool(task.decision.get("classification", {}).get("changes_code", True))
        mode = task.options.get("workspace") or rt.config.workspace.mode
        if not changes or mode == "canonical_readonly":
            return rt.workspaces.canonical(project)
        if mode == "direct":
            return await rt.workspaces.create_direct(project, task.id, None)
        return await rt.workspaces.create_isolated(project, task.id, None)

    async def _run_workflow(self, run: TaskRun) -> None:
        rt = self.rt
        state = run.state
        visits: dict[str, int] = state.setdefault("visits", {})
        while state.get("node"):
            run.cancel.raise_if_cancelled()
            task = rt.tasks.get(run.task.id)
            if task.cancel_requested_at is not None:
                raise OperationCancelled(task.cancel_reason or "cancelled")
            if task.pause_requested_at is not None:
                self._save(run)
                rt.tasks.end_attempt(run.attempt.id, AttemptStatus.SUSPENDED, fence=run.fence, usage=run.budget.snapshot())
                run.transition(TaskStatus.PAUSED, "paused by user at a node boundary")
                return
            node = run.workflow.nodes[state["node"]]
            visits[node.id] = visits.get(node.id, 0) + 1
            if visits[node.id] > node.max_visits:
                result = NodeResult("exhausted", f"node {node.id} visit limit ({node.max_visits}) reached")
            else:
                run.emit("node.started", node=node.id, kind=node.kind, role=node.role, visit=visits[node.id])
                started = time.monotonic()
                result = await self.handlers[node.kind](run, node)
                run.emit("node.completed", node=node.id, outcome=result.outcome, note=result.note,
                         duration_ms=int((time.monotonic() - started) * 1000))
            if result.note:
                run.notes.append(result.note)
            state.setdefault("history", []).append({"node": node.id, "outcome": result.outcome, "note": result.note})
            if node.kind == "finalize":
                state["node"] = None
                self._save(run)
                return
            if result.outcome == "suspended":
                return
            nxt = node.next.get(result.outcome) or node.next.get("*") or "finalize"
            state["node"] = nxt
            state.pop("suspended", None)
            self._save(run)

    def _save(self, run: TaskRun) -> None:
        self.rt.tasks.save_checkpoint(run.attempt.id, run.fence, run.state)
        self.rt.tasks.record_usage(run.attempt.id, run.budget.snapshot())

    # ================================================================ helpers
    def _route(self, run: TaskRun, role: str, *, independent_of: str | None = None, avoid: set[str] | None = None,
               context_tokens: int = 0) -> RouteDecision:
        cls = run.classification
        pins = run.task.options.get("pins") or {}
        pin = pins.get(role) or (run.task.options.get("model") if role in {"implementer", "corrector", "debugger", "planner", "researcher"} else None)
        decision = self.rt.router.route(RouteRequirements(
            role="implementer" if role == "corrector" else role, task_kind=cls.get("kind", "general"),
            needs_vision=bool(cls.get("signals", {}).get("ui")) and False, min_context=context_tokens,
            risk=cls.get("risk", "normal"), avoid_models=avoid or set(), independent_of=independent_of, pin=pin,
            extra_dimensions={"security": 0.6} if cls.get("risk") == "high" else {},
        ))
        run.emit("route.decision", **decision.to_dict())
        return decision

    async def _compile(self, run: TaskRun, stage: str, route: RouteDecision, **extra: Any) -> CompiledContext:
        rt = self.rt
        cfg = rt.config.context
        window = route.model.config.context_window
        budget = min(cfg.max_budget_tokens, int(window * cfg.budget_fraction))
        text = f"{run.task.title}\n{run.task.description}"
        skills = []
        if rt.skills is not None:
            skills = rt.skills.select(text=text, task_kind=run.classification.get("kind", ""), languages=list(run.profile.get("languages", {}))[:4],
                                      frameworks=[f.split(" ")[0] for f in run.profile.get("frameworks", [])], stage=stage)
            if skills:
                run.emit("skill.selected", stage=stage, skills=[{"name": s.skill.name, "version": s.skill.version, "reasons": s.reasons} for s in skills])
        memory_items = rt.memory.search(run.project.id, text, session_id=run.task.session_id, limit=rt.config.memory.max_items_in_context) \
            if rt.memory is not None and rt.config.memory.enabled else []
        if memory_items and rt.memory is not None:
            rt.memory.mark_used([m.id for m in memory_items])
        decisions = rt.decisions.relevant(run.project.id, text) if rt.decisions is not None else []
        focus = list(dict.fromkeys([*run.task.contract.get("focus_paths", []), *sorted(run.changed_paths)]))
        conversation = rt.recent_conversation(run.task)
        req = ContextRequest(
            stage=stage, project_id=run.project.id, root=run.workspace.path, task_title=run.task.title,
            task_description=run.task.description, budget_tokens=budget, contract=run.task.contract, profile=run.profile,
            profile_summary=run.profile.get("summary", ""), focus_paths=focus, plan=run.outputs.get("plan"),
            memory_items=memory_items, decisions=decisions, skills=skills,
            skill_catalog=rt.skills.catalog(exclude={s.skill.name for s in skills}) if rt.skills is not None else "",
            instruction_files=rt.config.context.instruction_files, conversation=conversation, **extra,
        )
        compiled = await rt.compiler.compile(req)
        snapshot_id = rt.record_context_snapshot(run.task.id, run.attempt.id, stage, compiled)
        run.emit("context.compiled", stage=stage, strategy=compiled.strategy, budget=compiled.budget, used=compiled.used,
                 included=len(compiled.included), excluded=len(compiled.excluded), snapshot_id=snapshot_id,
                 kinds=sorted({i.kind for i in compiled.included}))
        run.state["last_context_snapshot"] = snapshot_id
        return compiled

    async def _agent(self, run: TaskRun, node: Node, role: str, route: RouteDecision, compiled: CompiledContext, output_tool: str,
                     *, strategy: str | None = None, ctx: ToolContext | None = None) -> AgentOutcome:
        rt = self.rt
        ctx = ctx or run.ctx(role, node.id)
        extra = rt.extra_tools(role)
        tools = tools_for(role, extra, allow_spawn=run.task.depth < rt.config.budgets.max_subtask_depth)
        resume = run.state.get("suspended") if (run.state.get("suspended") or {}).get("node") == node.id else None
        messages = None
        pending: list[ToolCall] = []
        approval_id = None
        if resume:
            messages = [ChatMessage.from_dict(m) for m in rt.artifacts.read_json(resume["messages_artifact"], project_id=run.project.id)]
            pending = [ToolCall.from_dict(c) for c in resume.get("pending_calls", [])]
            approval_id = resume.get("approval_id")
            if resume.get("kind") == "input" and pending and pending[0].name == "ask_user":
                answer = rt.approvals.get(approval_id) if approval_id else None
                reply = (answer.reason or "(no answer)") if answer and answer.status == "approved" else "the user did not answer; proceed with best judgement"
                messages.append(ChatMessage("tool", f"user answered: {reply}", tool_call_id=pending[0].id, name="ask_user"))
                pending = pending[1:]
                approval_id = None
        spec = AgentSpec(role=role, node=node.id, system=system_prompt(role, strategy=strategy, output_tool=output_tool),
                         context_text=compiled.render(), tools=tools, route=route, output_tool=output_tool,
                         max_turns=rt.config.budgets.max_turns_per_node, max_structured_repairs=rt.config.budgets.max_structured_repairs,
                         compaction_threshold=rt.config.context.compaction_threshold, messages=messages, pending_calls=pending,
                         pending_approval=approval_id)
        artifact_key = f"conv:{node.id}"

        def checkpoint(msgs: list[ChatMessage], remaining: list[ToolCall]) -> None:
            art = rt.artifacts.put_json([m.to_dict() for m in msgs], kind="conversation", name=artifact_key, project_id=run.project.id,
                                        task_id=run.task.id, attempt_id=run.attempt.id)
            prev = run.state.get("conversations", {}).get(node.id)
            if prev:
                rt.db.execute("UPDATE artifacts SET state = 'expired' WHERE id = ?", (prev,))
            run.state.setdefault("conversations", {})[node.id] = art.id
            run.state["_pending_calls"] = [c.to_dict() for c in remaining]

        loop = AgentLoop(rt.client, rt.events, run.budget, checkpoint)
        call_ctx = CallContext(project_id=run.project.id, session_id=run.task.session_id, task_id=run.task.id, attempt_id=run.attempt.id,
                               node=node.id, role=role, purpose=strategy or node.kind, context_snapshot_id=run.state.get("last_context_snapshot"),
                               prompt_version=PROMPT_VERSION)
        outcome = await loop.run(spec, ctx, call_ctx)
        if outcome.status == "suspended" and outcome.suspend is not None:
            sus = outcome.suspend
            run.state["suspended"] = {"node": node.id, "kind": sus.kind, "approval_id": sus.approval_id,
                                      "messages_artifact": run.state.get("conversations", {}).get(node.id),
                                      "pending_calls": [c.to_dict() for c in outcome.pending_calls], "question": sus.question}
            self._save(run)
            rt.tasks.end_attempt(run.attempt.id, AttemptStatus.SUSPENDED, fence=run.fence, usage=run.budget.snapshot())
            if sus.kind == "input":
                run.transition(TaskStatus.NEEDS_INPUT, f"question for the user: {one_line(sus.question or '', 200)}")
            elif run.task.status != TaskStatus.AWAITING_APPROVAL:
                run.transition(TaskStatus.AWAITING_APPROVAL, f"approval needed: {sus.reason}")
        return outcome

    def _result_text(self, payload: dict[str, Any] | None) -> str:
        if not payload:
            return ""
        parts = [payload.get("summary", "")]
        for key in ("files_changed", "tests_added_or_changed", "commands_run", "remaining_issues"):
            if payload.get(key):
                parts.append(f"{key}: " + "; ".join(str(x) for x in payload[key][:20]))
        return "\n".join(p for p in parts if p)

    def _latest_verification(self, run: TaskRun) -> list[tuple[str, str]]:
        report = run.outputs.get("verification") or {}
        items = []
        for r in report.get("results", []):
            text = f"{r['status'].upper()}: {r['summary']}"
            if r.get("output_excerpt") and r["status"] != "pass":
                text += f"\n```\n{r['output_excerpt']}\n```"
            items.append((f"{r['kind']}{(':' + r['name']) if r.get('name') else ''}", text))
        for note in report.get("notes", []):
            items.append(("note", note))
        return items

    # ================================================================ nodes
    async def _plan(self, run: TaskRun, node: Node) -> NodeResult:
        route = self._route(run, "planner")
        compiled = await self._compile(run, node.stage or "planning", route, failures=self._prior_failures(run))
        outcome = await self._agent(run, node, "planner", route, compiled, "submit_plan")
        if outcome.status == "suspended":
            return NodeResult("suspended")
        if outcome.payload:
            run.outputs["plan"] = outcome.payload
            art = self.rt.artifacts.put_json(outcome.payload, kind="plan", project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id)
            run.emit("plan.ready", artifact_id=art.id, approach=one_line(outcome.payload.get("approach", ""), 300),
                     steps=[s.get("title") for s in outcome.payload.get("steps", [])][:20])
            return NodeResult("ok")
        return NodeResult("fail", f"planner produced no plan ({outcome.note or outcome.status}); implementing without a plan")

    async def _propose(self, run: TaskRun, node: Node) -> NodeResult:
        n = self.rt.config.routing.max_parallel_proposals
        routes: list[RouteDecision] = []
        avoid: set[str] = set()
        for _ in range(n):
            try:
                r = self._route(run, "planner", avoid=avoid)
            except ProviderError:
                break
            if r.model.key in avoid:
                break
            routes.append(r)
            avoid.add(r.model.key)
        if not routes:
            routes = [self._route(run, "planner")]

        async def one(i: int, route: RouteDecision) -> dict[str, Any] | None:
            compiled = await self._compile(run, node.stage or "architecture", route)
            sub = Node(f"{node.id}_{i}", node.kind, node.role, node.stage)
            outcome = await self._agent(run, sub, "planner", route, compiled, "submit_plan", ctx=run.ctx("planner", sub.id))
            return {"model": route.model.ref, "plan": outcome.payload} if outcome.payload else None

        results = await asyncio.gather(*(one(i, r) for i, r in enumerate(routes)), return_exceptions=True)
        proposals = [r for r in results if isinstance(r, dict)]
        errors = [r for r in results if isinstance(r, BaseException)]
        for err in errors:
            if isinstance(err, (OperationCancelled, LeaseLostError)):
                raise err
        run.outputs["proposals"] = proposals
        run.emit("proposals.ready", count=len(proposals), models=[p["model"] for p in proposals])
        return NodeResult("ok" if proposals else "fail", None if proposals else "no proposals were produced")

    async def _synthesize(self, run: TaskRun, node: Node) -> NodeResult:
        proposals = run.outputs.get("proposals") or []
        if not proposals:
            return NodeResult("fail", "no proposals to synthesize")
        chosen = proposals[0]["plan"]
        alternatives = [p["plan"].get("approach", "") for p in proposals[1:]]
        if len(proposals) > 1:
            route = self._route(run, "planner")
            prior = [(f"Proposal {i + 1} ({p['model']})", self.rt.compiler._plan_text(p["plan"])) for i, p in enumerate(proposals)]
            compiled = await self._compile(run, "architecture", route, prior=prior)
            outcome = await self._agent(run, node, "planner", route, compiled, "submit_plan")
            if outcome.status == "suspended":
                return NodeResult("suspended")
            if outcome.payload:
                chosen = outcome.payload
                alternatives = [p["plan"].get("approach", "") for p in proposals if p["plan"].get("approach") != chosen.get("approach")]
        run.outputs["plan"] = chosen
        if self.rt.decisions is not None:
            adr = self.rt.decisions.add(run.project.id, title=f"Approach: {one_line(run.task.title, 80)}",
                                        context=f"Compared {len(proposals)} independent proposal(s) for task {run.task.id}.",
                                        decision=chosen.get("approach", ""), alternatives=alternatives, status="proposed",
                                        source="workflow:collaborative", task_id=run.task.id, session_id=run.task.session_id)
            run.emit("decision.proposed", decision_id=adr.id, title=adr.title)
        return NodeResult("ok")

    async def _implement(self, run: TaskRun, node: Node) -> NodeResult:
        rt = self.rt
        role = node.role or "implementer"
        if run.task.status != TaskStatus.RUNNING:
            run.transition(TaskStatus.RUNNING, f"{node.id} started")
        route = self._route(run, role)
        run.implementer_model = route.model.key
        run.state["implementer_model"] = route.model.key
        extra: dict[str, Any] = {}
        stage = node.stage or "implementation"
        if node.params.get("correction"):
            diff = await rt.workspaces.diff(run.workspace)
            findings = [self._finding_text(f) for f in rt.reviews.open_findings(run.task.id)][:25]
            extra.update(diff=diff.patch or None, findings=findings, evidence=self._latest_verification(run),
                         prior=[("Previous implementation result", self._result_text(run.outputs.get("result")))])
        if role == "debugger" or stage == "debugging":
            extra.update(hypotheses=self._hypotheses(run), failures=self._prior_failures(run),
                         evidence=[*self._latest_verification(run), *run.outputs.get("repro_evidence", [])])
        elif run.task.mode == "recovery" or run.attempt.resumed_from:
            extra.setdefault("failures", self._prior_failures(run))
            if run.workspace.base_ref and "diff" not in extra:
                existing = await rt.workspaces.diff(run.workspace)
                if not existing.empty:
                    extra["diff"] = existing.patch
        compiled = await self._compile(run, stage, route, **extra)
        outcome = await self._agent(run, node, role, route, compiled, "submit_result")
        if outcome.status == "suspended":
            return NodeResult("suspended")
        if outcome.payload:
            run.outputs["result"] = outcome.payload
            rt.evidence.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=run.workspace.id,
                               kind=EvidenceKind.MODEL_CLAIM, status="pass", trust=Trust.CLAIMED,
                               summary=one_line(outcome.payload.get("summary", ""), 500), provider=route.model.provider_id,
                               model=route.model.model_id, data={"claims": outcome.payload, "node": node.id}, fence=run.fence)
        diff = await rt.workspaces.diff(run.workspace)
        note = None if outcome.payload else f"{role} finished without a structured result ({outcome.note or outcome.status})"
        if diff.empty:
            return NodeResult("no_changes", note or "no changes were made")
        return NodeResult("ok", note)

    async def _verify(self, run: TaskRun, node: Node) -> NodeResult:
        rt = self.rt
        run.transition(TaskStatus.VERIFYING, "running verification")
        report = await rt.verifier.run(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace=run.workspace,
                                       requirements=[r for r in run.requirements if r.kind != EvidenceKind.REVIEW],
                                       profile=run.profile, contract=run.task.contract, cancel=run.cancel, fence=run.fence,
                                       extra_env=rt.workspace_env(run.workspace, run.profile))
        run.outputs["verification"] = report.to_dict()
        required = {(r.kind, r.name) for r in run.requirements if r.required}
        failed = [r for r in report.results if (r.kind, r.name) in required and r.status != "pass"]
        run.transition(TaskStatus.RUNNING, "verification finished")
        summary = "; ".join(f"{r.kind}: {r.status}" for r in report.results)
        run.emit("verification.completed", ok=not failed, summary=summary, notes=report.notes)
        if failed:
            for r in failed:
                rt.failures.record(category="verification", error_class=f"{r.kind}_{r.status}", summary=r.summary, project_id=run.project.id,
                                   task_id=run.task.id, attempt_id=run.attempt.id, stage="verify")
            return NodeResult("fail", "verification failed: " + ", ".join(f"{r.kind} ({r.summary})" for r in failed[:4]))
        return NodeResult("ok")

    async def _review(self, run: TaskRun, node: Node) -> NodeResult:
        rt = self.rt
        run.transition(TaskStatus.REVIEWING, "independent review")
        diff = await rt.workspaces.diff(run.workspace)
        static = static_review(diff, result=run.outputs.get("result"), commands_run=run.commands_run, contract=run.task.contract,
                               redactor=rt.redactor)
        blocking = any(f.severity in {"critical", "high"} for f in static)
        review_ids = [rt.reviews.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, strategy="static",
                                        reviewer="deterministic", verdict="request_changes" if blocking else "approve",
                                        summary=f"{len(static)} deterministic finding(s)", findings=static, diff_hash=diff.diff_hash,
                                        fingerprint=diff.fingerprint, workspace_root=run.workspace.path).id]
        verdicts: list[str] = []
        independence = None
        if rt.config.verification.require_review:
            strategies = run.classification.get("review_strategies") or ["correctness"]
            for strategy in strategies:
                try:
                    route = self._route(run, "reviewer", independent_of=run.implementer_model or run.state.get("implementer_model"))
                except ProviderError as exc:
                    run.notes.append(f"review ({strategy}) could not be routed: {exc.message}")
                    verdicts.append("inconclusive")
                    continue
                independence = route.independence
                compiled = await self._compile(run, "security" if strategy == "security" else "review", route, diff=diff.patch,
                                               evidence=self._latest_verification(run),
                                               prior=[("Implementer's claimed result (verify, do not trust)", self._result_text(run.outputs.get("result")))])
                sub = Node(f"{node.id}_{strategy}", node.kind, "reviewer", node.stage)
                outcome = await self._agent(run, sub, "reviewer", route, compiled, "submit_review", strategy=strategy,
                                            ctx=run.ctx("reviewer", sub.id))
                if outcome.status == "suspended":
                    return NodeResult("suspended")
                if not outcome.payload:
                    verdicts.append("inconclusive")
                    rec = rt.reviews.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, strategy=strategy,
                                            reviewer=route.model.ref, verdict="inconclusive", summary=outcome.note or "reviewer produced no result",
                                            findings=[], diff_hash=diff.diff_hash, fingerprint=diff.fingerprint,
                                            workspace_root=run.workspace.path, independence=route.independence)
                    review_ids.append(rec.id)
                    continue
                payload = outcome.payload
                findings = [Finding(f["severity"], f["category"], f["title"], f["rationale"], f.get("file"), f.get("line"),
                                    f.get("evidence", ""), f.get("remediation", "")) for f in payload.get("findings", [])]
                verdicts.append(payload["verdict"])
                rec = rt.reviews.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, strategy=strategy,
                                        reviewer=route.model.ref, verdict=payload["verdict"], summary=payload.get("summary", ""),
                                        findings=findings, diff_hash=diff.diff_hash, fingerprint=diff.fingerprint,
                                        workspace_root=run.workspace.path, independence=route.independence)
                review_ids.append(rec.id)
        rt.reviews.supersede_older(run.task.id, diff.diff_hash)
        open_findings = rt.reviews.open_findings(run.task.id, diff_hash=diff.diff_hash)
        verdict = verdict_for(open_findings, verdicts)
        counts: dict[str, int] = {}
        for f in open_findings:
            counts[f["severity"]] = counts.get(f["severity"], 0) + 1
        status = {"approve": "pass", "request_changes": "fail"}.get(verdict, "inconclusive")
        rt.evidence.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=run.workspace.id,
                           kind=EvidenceKind.REVIEW, status=status, trust=Trust.OBSERVED,
                           summary=f"review {verdict}: " + (", ".join(f"{n} {s}" for s, n in counts.items()) or "no findings"),
                           fingerprint=diff.fingerprint, diff_hash=diff.diff_hash, fence=run.fence,
                           data={"verdict": verdict, "reviews": review_ids, "findings": counts, "independence": independence,
                                 "model_verdicts": verdicts})
        run.transition(TaskStatus.RUNNING, f"review {verdict}")
        if verdict == "request_changes":
            rt.failures.record(category="review", error_class="changes_requested", summary="; ".join(f["title"] for f in open_findings[:5]),
                               project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, stage="review")
        return NodeResult(verdict)

    async def _answer(self, run: TaskRun, node: Node) -> NodeResult:
        rt = self.rt
        route = self._route(run, "researcher")
        compiled = await self._compile(run, node.stage or "question", route)
        ctx = run.ctx("researcher", node.id)
        outcome = await self._agent(run, node, "researcher", route, compiled, "submit_answer", ctx=ctx)
        if outcome.status == "suspended":
            return NodeResult("suspended")
        payload = outcome.payload
        common = {"project_id": run.project.id, "task_id": run.task.id, "attempt_id": run.attempt.id, "workspace_id": run.workspace.id,
                  "fence": run.fence}
        mutations = int(rt.db.scalar("SELECT COUNT(*) FROM tool_calls WHERE attempt_id = ? AND status = 'ok' AND side_effect NOT IN ('none') "
                                     "AND side_effect IS NOT NULL", (run.attempt.id,)) or 0)
        rt.evidence.record(kind=EvidenceKind.WORKSPACE_UNCHANGED, status="pass" if mutations == 0 else "fail", trust=Trust.VERIFIED,
                           summary="no mutating tool calls were executed" if mutations == 0 else f"{mutations} mutating tool call(s) executed",
                           data={"mutating_calls": mutations}, **common)
        if not payload:
            return NodeResult("fail", outcome.note or "no answer was produced")
        run.outputs["answer"] = payload
        valid, invalid = [], []
        for c in payload.get("citations", []):
            try:
                path = resolve_within(run.workspace.path, c["path"])
                rel = path.relative_to(run.workspace.path.resolve()).as_posix()
            except (PathViolation, ValueError):
                invalid.append(f"{c['path']} (outside workspace)")
                continue
            if not path.is_file():
                invalid.append(f"{c['path']} (does not exist)")
            elif rel not in run.files_read:
                invalid.append(f"{c['path']} (cited but never read)")
            else:
                valid.append(rel)
        ok = bool(valid) and not invalid
        rt.evidence.record(kind=EvidenceKind.CITATIONS, status="pass" if ok else "fail", trust=Trust.VERIFIED,
                           summary=f"{len(valid)} valid citation(s)" + (f"; invalid: {', '.join(invalid[:5])}" if invalid else "") +
                           ("" if valid else "; answer has no verifiable citations"), data={"valid": valid, "invalid": invalid}, **common)
        art = rt.artifacts.put_text(payload.get("answer", ""), kind="answer", project_id=run.project.id, task_id=run.task.id,
                                    attempt_id=run.attempt.id, media_type="text/markdown")
        rt.evidence.record(kind=EvidenceKind.ANSWER, status="pass", trust=Trust.CLAIMED, summary=one_line(payload.get("answer", ""), 300),
                           artifact_id=art.id, provider=route.model.provider_id, model=route.model.model_id, **common)
        return NodeResult("ok")

    async def _review_existing(self, run: TaskRun, node: Node) -> NodeResult:
        rt = self.rt
        from coremain.workspaces.git import Git
        from coremain.workspaces.manager import FileChange, WorkspaceDiff

        ref = run.task.contract.get("review_ref") or "HEAD"
        git = Git(Path(run.project.root_path))
        res = await git.run("diff", "--no-color", "--binary", ref, check=False)
        patch = res.stdout.decode("utf-8", errors="replace") if res.code == 0 else ""
        names = await git.run("diff", "--name-status", ref, check=False)
        files = [FileChange(line.split("\t")[0][:1], line.split("\t")[-1]) for line in names.text.splitlines() if "\t" in line]
        from coremain.util.jsonutil import sha256_hex

        diff = WorkspaceDiff(files, patch, sha256_hex(patch), "", {"files": len(files)})
        if not patch.strip():
            run.outputs["answer"] = {"answer": f"There are no changes relative to `{ref}` to review.", "citations": []}
            rt.evidence.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=run.workspace.id,
                               kind=EvidenceKind.REVIEW, status="inconclusive", trust=Trust.OBSERVED, summary="nothing to review",
                               fence=run.fence)
            return NodeResult("ok", "empty diff")
        static = static_review(diff, result=None, commands_run=[], contract=run.task.contract, redactor=rt.redactor)
        rt.reviews.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, strategy="static", reviewer="deterministic",
                          verdict="request_changes" if any(f.severity in {"critical", "high"} for f in static) else "approve",
                          summary=f"{len(static)} deterministic finding(s)", findings=static, diff_hash=diff.diff_hash, fingerprint=None,
                          workspace_root=Path(run.project.root_path))
        verdicts = []
        for strategy in run.classification.get("review_strategies") or ["correctness"]:
            route = self._route(run, "reviewer")
            compiled = await self._compile(run, "security" if strategy == "security" else "review", route, diff=patch)
            sub = Node(f"{node.id}_{strategy}", node.kind, "reviewer", node.stage)
            outcome = await self._agent(run, sub, "reviewer", route, compiled, "submit_review", strategy=strategy, ctx=run.ctx("reviewer", sub.id))
            if outcome.status == "suspended":
                return NodeResult("suspended")
            payload = outcome.payload or {"verdict": "inconclusive", "summary": outcome.note or "no review produced", "findings": []}
            verdicts.append(payload["verdict"])
            findings = [Finding(f["severity"], f["category"], f["title"], f["rationale"], f.get("file"), f.get("line"), f.get("evidence", ""),
                                f.get("remediation", "")) for f in payload.get("findings", [])]
            rt.reviews.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, strategy=strategy,
                              reviewer=route.model.ref, verdict=payload["verdict"], summary=payload.get("summary", ""), findings=findings,
                              diff_hash=diff.diff_hash, fingerprint=None, workspace_root=Path(run.project.root_path))
        open_findings = rt.reviews.open_findings(run.task.id, diff_hash=diff.diff_hash)
        verdict = verdict_for(open_findings, verdicts)
        lines = [f"Review of changes relative to `{ref}` ({len(files)} files): **{verdict}**", ""]
        for f in open_findings:
            loc = f"{f['file']}:{f['line']}" if f.get("file") and f.get("line") else (f.get("file") or "general")
            lines.append(f"- **{f['severity']}** [{f['category']}] {f['title']} ({loc}) — {f['rationale']}"
                         + (f" Remediation: {f['remediation']}" if f.get("remediation") else ""))
        if not open_findings:
            lines.append("No findings.")
        run.outputs["answer"] = {"answer": "\n".join(lines), "citations": []}
        completed = any(v != "inconclusive" for v in verdicts)
        rt.evidence.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=run.workspace.id,
                           kind=EvidenceKind.REVIEW, status="pass" if completed else "inconclusive", trust=Trust.OBSERVED,
                           summary=f"structured review completed ({verdict}, {len(open_findings)} open finding(s))", diff_hash=diff.diff_hash,
                           data={"verdict": verdict}, fence=run.fence)
        mutations = int(rt.db.scalar("SELECT COUNT(*) FROM tool_calls WHERE attempt_id = ? AND status = 'ok' AND side_effect NOT IN ('none') "
                                     "AND side_effect IS NOT NULL", (run.attempt.id,)) or 0)
        rt.evidence.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=run.workspace.id,
                           kind=EvidenceKind.WORKSPACE_UNCHANGED, status="pass" if mutations == 0 else "fail", trust=Trust.VERIFIED,
                           summary=f"{mutations} mutating tool call(s)", fence=run.fence)
        return NodeResult("ok")

    async def _reproduce(self, run: TaskRun, node: Node) -> NodeResult:
        rt = self.rt
        cmd = run.task.contract.get("repro_command")
        test_cmd = (run.profile.get("commands") or {}).get("test") or rt.config.verification.commands.get("test")
        if not cmd and test_cmd:
            from coremain.intel.languages import is_test_path

            targets = [m.group(1) for m in re.finditer(r"([\w./-]+\.py(?:::[\w\[\]-]+)*)", run.task.description)
                       if is_test_path(m.group(1).split("::")[0]) and (run.workspace.path / m.group(1).split("::")[0]).exists()]
            if targets:
                cmd = f"{test_cmd} {targets[0]}"
            elif re.search(r"\b(failing|fails|failed) tests?\b|\btests?\b.{0,40}\b(failing|fails|failed|broken)\b",
                           run.task.description, re.I):
                cmd = test_cmd
        if not cmd:
            return NodeResult("skip", "no reproduction command identified; the debugger will reproduce manually")
        import os

        from coremain.security.commands import analyze_command
        from coremain.security.env import build_subprocess_env
        from coremain.security.policy import Capability, PolicyRequest

        decision = rt.policy.evaluate(PolicyRequest(capability=Capability.EXEC, target=cmd, workspace_root=run.workspace.path,
                                                    workspace_isolated=run.workspace.isolated, analysis=analyze_command(cmd, workspace=run.workspace.path),
                                                    project_id=run.project.id, task_id=run.task.id, tool="reproduce"))
        if decision.decision != "allow":
            return NodeResult("skip", f"reproduction command not allowed by policy ({decision.reason})")
        env = build_subprocess_env(os.environ, passthrough=rt.config.permissions.env_passthrough,
                                   extra={"CI": "1", **rt.workspace_env(run.workspace, run.profile)})
        result = await rt.processes.run(shell_command=cmd, cwd=run.workspace.path, env=env, timeout_s=float(rt.config.verification.timeout_s),
                                        cancel=run.cancel.child(), task_id=run.task.id, attempt_id=run.attempt.id)
        # pytest exit code 5 means "no tests collected", which is not a reproduction.
        reproduced = result.exit_code not in (0, None, 5)
        output = result.combined(8000)
        fp = await rt.workspaces.fingerprint(run.workspace)
        rt.evidence.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=run.workspace.id,
                           kind=EvidenceKind.REPRO, status="pass" if reproduced else "fail", trust=Trust.OBSERVED,
                           summary=("failure reproduced" if reproduced else "could not reproduce (command succeeded)") + f": `{cmd}`",
                           command=cmd, exit_code=result.exit_code, fingerprint=fp, fence=run.fence)
        run.outputs["repro_evidence"] = [("reproduction", f"`{cmd}` exited {result.exit_code}\n```\n{output[-6000:]}\n```")]
        return NodeResult("ok" if reproduced else "fail", None if reproduced else "failure not reproduced before changes")

    async def _finalize(self, run: TaskRun, node: Node) -> NodeResult:
        rt = self.rt
        ws = run.workspace
        diff = await rt.workspaces.diff(ws) if ws.base_ref else None
        fingerprint = diff.fingerprint if diff else (await rt.workspaces.fingerprint(ws) if run.changes_code else None)
        gate_ctx = GateContext(task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=ws.id, fingerprint=fingerprint,
                               requirements=run.requirements, min_level=rt.config.verification.min_level if run.changes_code else "weak",
                               diff_hash=diff.diff_hash if diff else None, notes=list(run.notes))
        with rt.db.tx() as conn:
            gate = rt.evidence.evaluate(conn, gate_ctx)
        rt.evidence.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=ws.id, kind=EvidenceKind.GATE,
                           status="pass" if gate.passed else "fail", trust=Trust.VERIFIED, summary=gate.summary, fingerprint=fingerprint,
                           data=gate.to_dict(), fence=run.fence)
        apply_info: dict[str, Any] | None = None
        auto_apply = run.task.options.get("auto_apply", rt.config.workspace.auto_apply)
        if gate.passed and run.changes_code and ws.isolated and auto_apply and diff is not None and not diff.empty:
            try:
                result = await rt.workspaces.apply_to_canonical(ws, run.project)
                apply_info = result.to_dict()
                rt.evidence.record(project_id=run.project.id, task_id=run.task.id, attempt_id=run.attempt.id, workspace_id=ws.id,
                                   kind=EvidenceKind.APPLY, status="pass", trust=Trust.VERIFIED,
                                   summary=f"applied {len(result.applied)} file(s), merged {len(result.merged)}, already present {len(result.already)}",
                                   data=apply_info, fence=run.fence)
                if result.merged:
                    run.notes.append(f"merged with concurrent edits in {', '.join(result.merged)}; re-run tests in your working tree")
            except WorkspaceConflictError as exc:
                rt.workspaces.set_status(ws, "preserved", reason="apply conflict")
                summary = f"{exc.message}. Changes are preserved in {ws.path}."
                report = render_report(run, gate, apply_info=None, final_status="needs_input", conflict=exc.details)
                self._final_message(run, report)
                rt.tasks.end_attempt(run.attempt.id, AttemptStatus.SUCCEEDED, fence=run.fence, usage=run.budget.snapshot())
                run.transition(TaskStatus.NEEDS_INPUT, summary, result_summary=summary)
                return NodeResult("conflict", summary)
        if gate.passed:
            summary = self._summary(run)
            final = "completed"
            report = render_report(run, gate, apply_info=apply_info, final_status=final)
            self._final_message(run, report)
            rt.tasks.end_attempt(run.attempt.id, AttemptStatus.SUCCEEDED, fence=run.fence, usage=run.budget.snapshot())
            try:
                run.transition(TaskStatus.COMPLETED, "evidence gate passed", result_summary=summary,
                               gate=lambda conn, task: rt.evidence.evaluate(conn, gate_ctx))
            except GateFailedError as exc:
                run.transition(TaskStatus.INCOMPLETE, f"gate failed at commit: {exc.message}", result_summary=summary)
                return NodeResult("incomplete", exc.message)
            if ws.isolated and ws.status == "active":
                rt.workspaces.set_status(ws, "applied" if apply_info else "released")
            rt.failures.resolve_for_task(run.task.id, resolution=summary, evidence_id=None, verified=True)
            rt.learn_from_success(run, gate)
            return NodeResult("completed")
        progress = bool(diff and not diff.empty) or bool(run.outputs.get("answer"))
        final_status = TaskStatus.INCOMPLETE if progress else TaskStatus.FAILED
        summary = f"{self._summary(run) or 'No verified result'} — gate: {gate.summary}"
        report = render_report(run, gate, apply_info=None, final_status=final_status.value)
        self._final_message(run, report)
        rt.tasks.end_attempt(run.attempt.id, AttemptStatus.SUCCEEDED if progress else AttemptStatus.FAILED, fence=run.fence,
                             error_class=None if progress else "no_result", usage=run.budget.snapshot())
        if ws.isolated and ws.status == "active":
            rt.workspaces.set_status(ws, "preserved" if progress else "released", reason="task incomplete" if progress else "no changes")
        run.transition(final_status, gate.summary, result_summary=summary)
        return NodeResult(final_status.value, gate.summary)

    # ================================================================ misc helpers
    def _summary(self, run: TaskRun) -> str:
        if run.outputs.get("answer"):
            return run.outputs["answer"].get("answer", "")[:4000]
        result = run.outputs.get("result") or {}
        return result.get("summary", "")[:2000]

    def _final_message(self, run: TaskRun, report: str) -> None:
        if run.task.session_id:
            self.rt.sessions.add_message(run.task.session_id, "assistant", report, task_id=run.task.id,
                                         meta={"kind": "task_report", "status": run.task.status.value})

    @staticmethod
    def _finding_text(f: dict[str, Any]) -> str:
        loc = f"{f['file']}:{f['line']}" if f.get("file") and f.get("line") else (f.get("file") or "")
        return (f"[{f['severity']}/{f['category']}] {f['title']} {('@ ' + loc) if loc else ''}\nRationale: {f['rationale']}"
                + (f"\nEvidence: {f['evidence']}" if f.get("evidence") else "") + (f"\nRemediation: {f['remediation']}" if f.get("remediation") else ""))

    def _hypotheses(self, run: TaskRun) -> list[str]:
        rows = self.rt.db.query("SELECT statement, status FROM debug_hypotheses WHERE task_id = ? ORDER BY created_at", (run.task.id,))
        return [f"[{r['status']}] {r['statement']}" for r in rows]

    def _prior_failures(self, run: TaskRun) -> list[str]:
        rows = self.rt.db.query("SELECT stage, category, error_class, summary FROM failures WHERE task_id = ? ORDER BY created_at DESC LIMIT 8",
                                (run.task.id,))
        out = [f"{r['stage'] or r['category']}: {r['error_class']} — {r['summary']}" for r in rows]
        for a in self.rt.tasks.attempts(run.task.id):
            if a.id != run.attempt.id and a.error_message:
                out.append(f"attempt {a.number} ended {a.status.value}: {a.error_message}")
        return out

    async def _handle_error(self, run: TaskRun, exc: BaseException) -> None:
        rt = self.rt
        task_id = run.task.id
        usage = run.budget.snapshot()

        def safe(fn: Any) -> None:
            try:
                fn()
            except LeaseLostError:
                log.warning("lease lost while recording failure for %s", task_id)
            except CoreError as err:
                log.warning("could not record failure state for %s: %s", task_id, err)

        if isinstance(exc, OperationCancelled):
            safe(lambda: rt.tasks.end_attempt(run.attempt.id, AttemptStatus.CANCELLED, fence=run.fence, error_class="cancelled",
                                              error_message=exc.message, usage=usage))
            if run.workspace.isolated:
                safe(lambda: rt.workspaces.set_status(run.workspace, "preserved", reason="task cancelled"))
            safe(lambda: rt.approvals.cancel_for_task(task_id))
            current = rt.tasks.get(task_id)
            if current.cancel_requested_at is not None:
                safe(lambda: rt.tasks.transition(task_id, TaskStatus.CANCELLED, reason=exc.message, fence=run.fence))
            else:
                safe(lambda: rt.tasks.transition(task_id, TaskStatus.INTERRUPTED, reason=f"interrupted: {exc.message}", fence=run.fence))
            return
        if isinstance(exc, ProviderError):
            rt.failures.record(category="provider", error_class=exc.error_class.value, summary=exc.message, project_id=run.project.id,
                               task_id=task_id, attempt_id=run.attempt.id, stage=run.state.get("node"),
                               context={"provider": exc.provider_id, "model": exc.model_id, "status": exc.status_code})
            safe(lambda: rt.tasks.end_attempt(run.attempt.id, AttemptStatus.FAILED, fence=run.fence, error_class=exc.error_class.value,
                                              error_message=exc.message, usage=usage))
            safe(lambda: self._save(run))
            reason = exc.block_reason or ("provider_unavailable" if exc.transient else None)
            if run.workspace.isolated:
                safe(lambda: rt.workspaces.set_status(run.workspace, "preserved", reason=f"provider error: {exc.error_class.value}"))
            message = f"{exc.message}" + (f" — {exc.hint}" if exc.hint else "")
            if reason:
                safe(lambda: rt.tasks.transition(task_id, TaskStatus.BLOCKED, reason=message, block_reason=reason, fence=run.fence))
            else:
                safe(lambda: rt.tasks.transition(task_id, TaskStatus.FAILED, reason=message, fence=run.fence))
            self._final_error_message(run, f"Provider error ({exc.error_class.value}): {message}")
            return
        if isinstance(exc, BudgetExceededError):
            rt.failures.record(category="workflow", error_class="budget_exceeded", summary=exc.message, project_id=run.project.id,
                               task_id=task_id, attempt_id=run.attempt.id, stage=run.state.get("node"))
            safe(lambda: rt.tasks.end_attempt(run.attempt.id, AttemptStatus.FAILED, fence=run.fence, error_class="budget_exceeded",
                                              error_message=exc.message, usage=usage))
            diff = await rt.workspaces.diff(run.workspace) if run.workspace.base_ref else None
            status = TaskStatus.INCOMPLETE if diff is not None and not diff.empty else TaskStatus.FAILED
            if run.workspace.isolated:
                safe(lambda: rt.workspaces.set_status(run.workspace, "preserved", reason="budget exhausted"))
            safe(lambda: rt.tasks.transition(task_id, status, reason=exc.message, fence=run.fence))
            self._final_error_message(run, f"Stopped: {exc.message}. Partial work is preserved in the task workspace.")
            return
        mutating_node = (run.state.get("node") or "") in {"implement", "correct", "investigate"}
        category = "workspace" if isinstance(exc, WorkspaceError) else "internal"
        if not isinstance(exc, CoreError):
            log.error("internal error in task %s", task_id, exc_info=exc)
        message = f"{type(exc).__name__}: {exc}"
        rt.failures.record(category=category, error_class=getattr(exc, "code", "internal_error"), summary=message, project_id=run.project.id,
                           task_id=task_id, attempt_id=run.attempt.id, stage=run.state.get("node"))
        safe(lambda: rt.tasks.end_attempt(run.attempt.id, AttemptStatus.FAILED, fence=run.fence, error_class=getattr(exc, "code", "internal_error"),
                                          error_message=message[:1000], usage=usage))
        if run.workspace.isolated:
            safe(lambda: rt.workspaces.set_status(run.workspace, "preserved", reason="unexpected error"))
        target = TaskStatus.UNKNOWN if mutating_node else TaskStatus.FAILED
        safe(lambda: rt.tasks.transition(task_id, target, reason=f"unexpected error: {message[:300]}", fence=run.fence))
        self._final_error_message(run, f"Unexpected error during {run.state.get('node')}: {message[:500]}")

    def _final_error_message(self, run: TaskRun, text: str) -> None:
        if run.task.session_id:
            with contextlib.suppress(CoreError):
                self.rt.sessions.add_message(run.task.session_id, "assistant", f"**Task {run.task.id} stopped.** {text}", task_id=run.task.id,
                                             meta={"kind": "task_error"})


def estimate_context_need(text: str) -> int:
    return estimate_tokens(text) + 8000
