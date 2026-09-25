"""Structured-contribution tools.

Model contributions are persisted as structured artifacts (plans, results, answers, reviews,
hypotheses, decisions) rather than free-form chatter. ``submit_*`` tools end the agent loop;
their payloads are validated by the executor against these schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from coremain.errors import ToolError
from coremain.security.policy import Capability
from coremain.tools.base import Tool, ToolContext, ToolInput, ToolResult
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, loads


class PlanStep(ToolInput):
    title: str
    detail: str = ""
    files: list[str] = Field(default_factory=list)


class DecisionInput(ToolInput):
    title: str
    decision: str
    alternatives: list[str] = Field(default_factory=list)
    rationale: str = ""


class AcceptanceItem(ToolInput):
    criterion: str
    verification: str


class SubmitPlan(Tool):
    name = "submit_plan"
    description = "Submit the implementation plan (ends planning). Include concrete steps, files, risks and how each acceptance criterion will be verified."
    capability = Capability.CONTROL
    terminal = True

    class Input(ToolInput):
        approach: str = Field(description="One-paragraph summary of the approach")
        steps: list[PlanStep] = Field(min_length=1)
        risks: list[str] = Field(default_factory=list)
        test_strategy: str = ""
        acceptance: list[AcceptanceItem] = Field(default_factory=list)
        decisions: list[DecisionInput] = Field(default_factory=list)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        payload = args.model_dump()
        ctx.structured[self.name] = payload
        return ToolResult(True, "plan recorded", terminal=True, payload=payload)


class SubmitResult(Tool):
    name = "submit_result"
    description = ("Submit the implementation result (ends implementation). Report exactly what changed and what you ran; "
                   "claims are checked against the actual diff and command records.")
    capability = Capability.CONTROL
    terminal = True

    class Input(ToolInput):
        summary: str
        files_changed: list[str] = Field(default_factory=list)
        tests_added_or_changed: list[str] = Field(default_factory=list)
        commands_run: list[str] = Field(default_factory=list)
        remaining_issues: list[str] = Field(default_factory=list)
        confidence: Literal["low", "medium", "high"] = "medium"

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        payload = args.model_dump()
        ctx.structured[self.name] = payload
        return ToolResult(True, "result recorded", terminal=True, payload=payload)


class Citation(ToolInput):
    path: str
    start_line: int | None = None
    end_line: int | None = None
    note: str = ""


class SubmitAnswer(Tool):
    name = "submit_answer"
    description = "Submit the final answer for a question/research task, with citations to files you actually read."
    capability = Capability.CONTROL
    terminal = True

    class Input(ToolInput):
        answer: str = Field(min_length=1, description="Markdown answer")
        citations: list[Citation] = Field(default_factory=list)
        confidence: Literal["low", "medium", "high"] = "medium"
        open_questions: list[str] = Field(default_factory=list)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        payload = args.model_dump()
        ctx.structured[self.name] = payload
        return ToolResult(True, "answer recorded", terminal=True, payload=payload)


class FindingInput(ToolInput):
    severity: Literal["critical", "high", "medium", "low", "info"]
    category: Literal["correctness", "security", "concurrency", "edge_case", "architecture", "tests", "performance",
                      "dependency", "misleading_claim", "maintainability", "requirements"]
    title: str
    file: str | None = None
    line: int | None = None
    rationale: str
    evidence: str = ""
    remediation: str = ""


class SubmitReview(Tool):
    name = "submit_review"
    description = ("Submit the review (ends review). Verdict 'approve' only if no blocking problems remain. Every finding needs "
                   "severity, location, rationale, evidence and a concrete remediation.")
    capability = Capability.CONTROL
    terminal = True

    class Input(ToolInput):
        verdict: Literal["approve", "request_changes", "inconclusive"]
        summary: str
        findings: list[FindingInput] = Field(default_factory=list)
        criteria_checked: list[str] = Field(default_factory=list)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        payload = args.model_dump()
        ctx.structured[self.name] = payload
        return ToolResult(True, "review recorded", terminal=True, payload=payload)


class RecordHypothesis(Tool):
    name = "record_hypothesis"
    description = "Record a debugging hypothesis before testing it. Returns an id to use with record_experiment."
    capability = Capability.CONTROL

    class Input(ToolInput):
        statement: str = Field(min_length=5)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        if not ctx.task_id:
            raise ToolError("hypotheses require a task", error_class="no_task")
        db = ctx.services.db
        now = ctx.services.events.clock.now()
        hyp_id = new_id("hyp", now=now)
        db.execute("INSERT INTO debug_hypotheses(id, task_id, attempt_id, statement, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                   (hyp_id, ctx.task_id, ctx.attempt_id, args.statement, "open", now, now))
        ctx.services.events.emit("debug.hypothesis", project_id=ctx.project_id, task_id=ctx.task_id, attempt_id=ctx.attempt_id,
                                 data={"hypothesis_id": hyp_id, "statement": args.statement})
        return ToolResult(True, f"hypothesis {hyp_id} recorded", data={"id": hyp_id})


class RecordExperiment(Tool):
    name = "record_experiment"
    description = "Record the result of an experiment that tests a hypothesis (confirmed/refuted/inconclusive)."
    capability = Capability.CONTROL

    class Input(ToolInput):
        hypothesis_id: str
        action: str
        observation: str
        verdict: Literal["confirmed", "refuted", "inconclusive"]

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        db = ctx.services.db
        row = db.one("SELECT * FROM debug_hypotheses WHERE id = ? AND task_id = ?", (args.hypothesis_id, ctx.task_id))
        if row is None:
            raise ToolError(f"unknown hypothesis {args.hypothesis_id}", error_class="not_found")
        experiments = loads(row["experiments_json"], [])
        experiments.append({"action": args.action, "observation": args.observation[:2000], "verdict": args.verdict})
        status = {"confirmed": "confirmed", "refuted": "refuted"}.get(args.verdict, row["status"] if row["status"] != "open" else "inconclusive")
        db.execute("UPDATE debug_hypotheses SET experiments_json = ?, status = ?, updated_at = ? WHERE id = ?",
                   (dumps(experiments), status, ctx.services.events.clock.now(), args.hypothesis_id))
        ctx.services.events.emit("debug.experiment", project_id=ctx.project_id, task_id=ctx.task_id, attempt_id=ctx.attempt_id,
                                 data={"hypothesis_id": args.hypothesis_id, "verdict": args.verdict, "action": args.action[:200]})
        return ToolResult(True, f"experiment recorded; hypothesis now {status}")


class RecordDecision(Tool):
    name = "record_decision"
    description = "Propose an architecture decision record (context, decision, rejected alternatives, consequences). The user can accept it."
    capability = Capability.CONTROL

    class Input(ToolInput):
        title: str
        context: str
        decision: str
        alternatives: list[str] = Field(default_factory=list)
        consequences: str = ""

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        db = ctx.services.db
        now = ctx.services.events.clock.now()
        adr_id = new_id("adr", now=now)
        db.execute(
            "INSERT INTO decisions(id, project_id, session_id, task_id, title, status, context, decision, alternatives_json, consequences, "
            "source, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (adr_id, ctx.project_id, ctx.session_id, ctx.task_id, args.title, "proposed", args.context, args.decision,
             dumps(args.alternatives), args.consequences, f"model:{ctx.role}", now, now),
        )
        ctx.services.events.emit("decision.proposed", project_id=ctx.project_id, task_id=ctx.task_id, data={"decision_id": adr_id, "title": args.title})
        return ToolResult(True, f"decision {adr_id} proposed (status: proposed until accepted)", data={"id": adr_id})


class AskUser(Tool):
    name = "ask_user"
    description = "Ask the user a question when requirements are ambiguous or information is missing. Use sparingly."
    capability = Capability.CONTROL

    class Input(ToolInput):
        question: str = Field(min_length=5)
        context: str | None = None

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        from coremain.domain.states import TaskStatus
        from coremain.tools.executor import SuspendRequested

        handler = ctx.services.input_handler
        if handler is None:
            approval = ctx.services.approvals.request(
                capability="user_input", summary=args.question[:300], request={"question": args.question, "context": args.context},
                project_id=ctx.project_id, session_id=ctx.session_id, task_id=ctx.task_id, attempt_id=ctx.attempt_id, tool_call_id=None,
            )
            raise SuspendRequested("input", approval_id=approval.id, question=args.question, reason="question for the user")
        if ctx.task_id:
            ctx.services.tasks.transition(ctx.task_id, TaskStatus.NEEDS_INPUT, reason=args.question[:200], fence=ctx.fence)
        answer = await ctx.cancel.run(handler(args.question, args.context))
        if ctx.task_id:
            ctx.services.tasks.transition(ctx.task_id, TaskStatus.RUNNING, reason="user answered", fence=ctx.fence)
        return ToolResult(True, f"user answered: {answer}")


class SpawnSubtask(Tool):
    name = "spawn_subtask"
    description = ("Delegate an independent, well-scoped subtask (e.g. focused research) to another worker. Bounded by depth and "
                   "fan-out budgets. Returns the subtask id; its result is added to your context when it finishes.")
    capability = Capability.TASK_SPAWN

    class Input(ToolInput):
        title: str
        instructions: str
        kind: Literal["research", "question", "review"] = "research"

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        spawn = ctx.services.spawn_subtask
        if spawn is None:
            raise ToolError("subtask delegation is not available in this context", error_class="unavailable")
        result = await spawn(ctx, args.title, args.instructions, args.kind)
        return ToolResult(True, result)


CONTROL_TOOLS: list[type[Tool]] = [SubmitPlan, SubmitResult, SubmitAnswer, SubmitReview, RecordHypothesis, RecordExperiment,
                                   RecordDecision, AskUser, SpawnSubtask]
