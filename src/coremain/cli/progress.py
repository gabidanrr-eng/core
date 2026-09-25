"""Live progress rendering for `core run` (human lines or JSON lines)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import click

from coremain.cli.output import Output, styled_status
from coremain.events import Event, Subscription
from coremain.runtime.approvals import Approval
from coremain.tools.base import ApprovalDecision
from coremain.util.text import one_line


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
        k = ev.kind
        if k == "model.delta":
            if self.verbose:
                c.print(d.get("text", ""), end="", style="dim", markup=False, highlight=False)
                self._delta_open = True
            return
        if self._delta_open:
            c.print()
            self._delta_open = False
        if k == "mode.selected":
            c.print(f"[bold]▸ mode[/] {d.get('mode')} [dim]({'; '.join(d.get('reasons', [])[:3])})[/]")
            if d.get("requirements"):
                c.print(f"[dim]  completion requires: {', '.join(d['requirements'])}[/]")
        elif k == "route.decision":
            reasons = "; ".join(d.get("reasons", [])[:3])
            c.print(f"[dim]▸ {d.get('role')} → {d.get('model')} ({reasons})[/]")
        elif k == "node.started":
            c.print(
                f"[bold cyan]● {d.get('node')}[/]" + (f" [dim]({d.get('role')})[/]" if d.get("role") else "")
            )
        elif k == "node.completed" and d.get("note"):
            c.print(f"[dim]  ↳ {d.get('outcome')}: {one_line(str(d.get('note')), 160)}[/]")
        elif k == "tool.call":
            ok = d.get("ok")
            mark = "[green]✓[/]" if ok else ("[yellow]⊘[/]" if d.get("status") == "denied" else "[red]✗[/]")
            args = d.get("args") or {}
            target = (
                args.get("path")
                or args.get("command")
                or args.get("pattern")
                or args.get("query")
                or args.get("name")
                or ""
            )
            c.print(
                f"  {mark} {d.get('tool')} [dim]{one_line(str(target), 90)}[/]"
                + (f" [red]{one_line(str(d.get('summary', '')), 100)}[/]" if not ok else "")
            )
        elif k == "agent.message":
            c.print(f"  [italic dim]{one_line(str(d.get('text', '')), 180)}[/]")
        elif k == "context.compiled":
            c.print(
                f"[dim]  context: {d.get('used')}/{d.get('budget')} tokens · {', '.join(d.get('kinds', [])[:8])}[/]"
            )
        elif k == "skill.selected":
            c.print(f"[dim]  skills: {', '.join(s['name'] for s in d.get('skills', []))}[/]")
        elif k == "verification.completed":
            c.print(
                f"  {'[green]✓' if d.get('ok') else '[red]✗'} verification[/] [dim]{one_line(str(d.get('summary', '')), 160)}[/]"
            )
        elif k == "review.completed":
            verdict = d.get("verdict")
            style = "green" if verdict == "approve" else "yellow"
            c.print(
                f"  [{style}]review {verdict}[/] [dim]{d.get('strategy')} by {d.get('reviewer')} · {len(d.get('findings', []))} finding(s)[/]"
            )
        elif k == "task.transition":
            c.print(
                f"[dim]  status → [/]{styled_status(d.get('to', ''))} [dim]{one_line(str(d.get('reason', '')), 120)}[/]"
            )
        elif k == "route.fallback":
            c.print(
                f"[yellow]  fallback: {d.get('from')} → {d.get('to')} after {d.get('error_class')} (configured)[/]"
            )
        elif k == "model.retry":
            c.print(
                f"[yellow]  retrying {d.get('model')} after {d.get('error_class')} (retry {d.get('retry')}, {d.get('delay_s')}s)[/]"
            )
        elif k == "model.call.failed":
            c.print(
                f"[red]  model error ({d.get('error_class')}): {one_line(str(d.get('message', '')), 200)}[/]"
            )
            if d.get("hint"):
                c.print(f"[yellow]  hint: {d['hint']}[/]")
        elif k == "approval.requested":
            c.print(f"[bold yellow]  approval requested:[/] {d.get('summary')}")
        elif k == "workspace.created":
            c.print(f"[dim]  workspace: {d.get('kind')} {d.get('path', '')}[/]")
        elif k == "workspace.conflict":
            c.print("[red]  apply conflict with concurrent changes; workspace preserved[/]")
        elif k in {"tool.crashed", "failure.recorded"} and self.verbose:
            c.print(f"[red]  {k}: {one_line(str(d), 200)}[/]")

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
