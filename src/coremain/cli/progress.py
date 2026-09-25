"""Live progress rendering for `core run` (human lines or JSON lines)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import click

from coremain.cli.output import Output
from coremain.events import Event, Subscription
from coremain.runtime.approvals import Approval
from coremain.tools.base import ApprovalDecision
from coremain.ui.describe import describe_event


class ProgressPrinter:
    def __init__(self, output: Output, *, jsonl: bool, verbose: bool):
        self.out = output
        self.jsonl = jsonl
        self.verbose = verbose
        self._delta_open = False

    def handle(self, ev: Event) -> None:
        if self.jsonl:
            if ev.durable or self.verbose:
                self.out.jsonl(ev.to_dict())
            return
        c = self.out.console
        d = ev.data
        if ev.kind == "model.delta":
            if self.verbose:
                c.print(d.get("text", ""), end="", style="dim", markup=False, highlight=False)
                self._delta_open = True
            return
        if self._delta_open:
            c.print()
            self._delta_open = False
        line = describe_event(ev, verbose=self.verbose)
        if line is not None:
            c.print(line)

    async def consume(self, sub: Subscription) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            async for ev in sub:
                self.handle(ev)


async def terminal_approval(output: Output, approval: Approval) -> ApprovalDecision:
    req = approval.request
    output.console.print(f"\n[bold yellow]Approval needed[/] ({approval.capability}): {approval.summary}")
    if req.get("reason"):
        output.console.print(f"[dim]reason: {req['reason']}  risk: {req.get('risk') or 'n/a'}[/]")
    choice = await asyncio.to_thread(
        click.prompt, "Allow? [y]es once / [s]ession / [n]o", default="n", show_default=False
    )
    choice = str(choice).strip().lower()[:1]
    if choice == "y":
        return ApprovalDecision(True, "once")
    if choice == "s":
        return ApprovalDecision(True, "session")
    return ApprovalDecision(False, "once", "rejected at terminal prompt")


async def terminal_input(output: Output, question: str, context: str | None) -> str:
    output.console.print(f"\n[bold yellow]Question from the agent:[/] {question}")
    if context:
        output.console.print(f"[dim]{context}[/]")
    return str(await asyncio.to_thread(click.prompt, "Answer"))


def event_filter(task_ids: set[str]) -> Any:
    def predicate(ev: Event) -> bool:
        return ev.task_id in task_ids or (ev.kind == "task.created" and ev.data.get("parent") in task_ids)

    return predicate
