"""One-line, Rich-markup descriptions of runtime events, shared by the CLI and the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coremain.util.text import one_line

if TYPE_CHECKING:
    from coremain.events import Event

STATUS_STYLE = {
    "completed": "green",
    "running": "cyan",
    "verifying": "cyan",
    "reviewing": "cyan",
    "queued": "blue",
    "pending": "blue",
    "awaiting_approval": "yellow",
    "needs_input": "yellow",
    "blocked": "red",
    "paused": "yellow",
    "interrupted": "yellow",
    "incomplete": "yellow",
    "failed": "red",
    "cancelled": "dim",
    "unknown": "magenta",
}


def esc(text: object) -> str:
    return str(text).replace("[", r"\[")


def styled_status(status: str) -> str:
    return f"[{STATUS_STYLE.get(status, 'white')}]{status}[/]"


def tool_target(args: dict[str, object]) -> str:
    for key in ("path", "command", "pattern", "query", "name", "url", "library"):
        if args.get(key):
            return str(args[key])
    return ""


def describe_event(ev: Event, *, verbose: bool = False) -> str | None:
    """A markup line for ``ev`` or None when it is not worth showing."""
    d = ev.data
    k = ev.kind
    if k == "mode.selected":
        line = f"[bold]▸ mode[/] {esc(d.get('mode'))} [dim]({esc('; '.join(d.get('reasons', [])[:3]))})[/]"
        if d.get("requirements"):
            line += f"\n[dim]  completion requires: {esc(', '.join(d['requirements']))}[/]"
        return line
    if k == "route.decision":
        return f"[dim]▸ {esc(d.get('role'))} → {esc(d.get('model'))} ({esc('; '.join(d.get('reasons', [])[:3]))})[/]"
    if k == "node.started":
        return f"[bold cyan]● {esc(d.get('node'))}[/]" + (
            f" [dim]({esc(d.get('role'))})[/]" if d.get("role") else ""
        )
    if k == "node.completed":
        return (
            f"[dim]  ↳ {esc(d.get('outcome'))}: {esc(one_line(str(d.get('note')), 160))}[/]"
            if d.get("note")
            else None
        )
    if k == "tool.call":
        ok = d.get("ok")
        mark = "[green]✓[/]" if ok else ("[yellow]⊘[/]" if d.get("status") == "denied" else "[red]✗[/]")
        target = tool_target(d.get("args") or {})
        line = f"  {mark} {esc(d.get('tool'))} [dim]{esc(one_line(target, 90))}[/]"
        if not ok:
            line += f" [red]{esc(one_line(str(d.get('summary', '')), 100))}[/]"
        return line
    if k == "agent.message":
        return f"  [italic dim]{esc(one_line(str(d.get('text', '')), 180))}[/]"
    if k == "context.compiled":
        return f"[dim]  context: {d.get('used')}/{d.get('budget')} tokens · {esc(', '.join(d.get('kinds', [])[:8]))}[/]"
    if k == "skill.selected":
        return f"[dim]  skills: {esc(', '.join(s['name'] for s in d.get('skills', [])))}[/]"
    if k == "verification.completed":
        mark = "[green]✓" if d.get("ok") else "[red]✗"
        return f"  {mark} verification[/] [dim]{esc(one_line(str(d.get('summary', '')), 160))}[/]"
    if k == "review.completed":
        style = "green" if d.get("verdict") == "approve" else "yellow"
        return (
            f"  [{style}]review {esc(d.get('verdict'))}[/] [dim]{esc(d.get('strategy'))} by {esc(d.get('reviewer'))} · "
            f"{len(d.get('findings', []))} finding(s)[/]"
        )
    if k == "task.transition":
        return f"[dim]  status → [/]{styled_status(str(d.get('to', '')))} [dim]{esc(one_line(str(d.get('reason', '')), 120))}[/]"
    if k == "route.fallback":
        return f"[yellow]  fallback: {esc(d.get('from'))} → {esc(d.get('to'))} after {esc(d.get('error_class'))} (configured)[/]"
    if k == "model.retry":
        return (
            f"[yellow]  retrying {esc(d.get('model'))} after {esc(d.get('error_class'))} "
            f"(retry {d.get('retry')}, {d.get('delay_s')}s)[/]"
        )
    if k == "model.call.failed":
        line = f"[red]  model error ({esc(d.get('error_class'))}): {esc(one_line(str(d.get('message', '')), 200))}[/]"
        if d.get("hint"):
            line += f"\n[yellow]  hint: {esc(d['hint'])}[/]"
        return line
    if k == "approval.requested":
        return f"[bold yellow]  approval requested:[/] {esc(d.get('summary'))}"
    if k == "approval.decided":
        return f"[dim]  approval {esc(d.get('status'))} ({esc(d.get('scope'))})[/]"
    if k == "workspace.created":
        return f"[dim]  workspace: {esc(d.get('kind'))} {esc(d.get('path', ''))}[/]"
    if k == "workspace.applied":
        return f"[green]  applied to working tree[/] [dim]{esc(one_line(str(d.get('applied', '')), 120))}[/]"
    if k == "workspace.conflict":
        return "[red]  apply conflict with concurrent changes; workspace preserved[/]"
    if k == "plan.ready":
        return f"[dim]  plan: {len(d.get('steps', []))} step(s)[/]"
    if k in {"tool.crashed", "failure.recorded"} and verbose:
        return f"[red]  {k}: {esc(one_line(str(d), 200))}[/]"
    return None
