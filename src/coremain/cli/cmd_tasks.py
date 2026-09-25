"""`core session`, `core task`, `core approvals`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click

from coremain.cli.common import CLIContext, runtime_command
from coremain.cli.output import ago, styled_status, task_exit_code
from coremain.domain.states import TaskStatus
from coremain.errors import ConflictError, NotFoundError, UsageError
from coremain.util.jsonutil import dumps
from coremain.util.text import one_line


# ============================================================================ sessions
@click.group()
def session() -> None:
    """Engineering sessions (durable conversation + tasks + evidence)."""


@session.command("list")
@click.option("--all", "include_archived", is_flag=True, help="Include archived sessions.")
@runtime_command()
async def session_list(ctx: CLIContext, rt: Any, include_archived: bool) -> int:
    project = rt.require_project()
    rows = []
    for s in rt.sessions.list(project.id, include_archived=include_archived):
        n = rt.db.scalar("SELECT COUNT(*) FROM tasks WHERE session_id = ?", (s.id,))
        rows.append({**s.__dict__, "tasks": n})
    ctx.output.data(
        rows,
        lambda c: ctx.output.table(
            ["id", "title", "status", "tasks", "updated", "parent"],
            [
                [
                    r["id"],
                    one_line(r["title"], 60),
                    r["status"],
                    r["tasks"],
                    ago(r["updated_at"]),
                    r["parent_session_id"] or "",
                ]
                for r in rows
            ],
        ),
    )
    return 0


@session.command("new")
@click.argument("title", required=False, default="New session")
@runtime_command()
async def session_new(ctx: CLIContext, rt: Any, title: str) -> int:
    s = rt.sessions.create(rt.require_project().id, title)
    ctx.output.data(s.__dict__, f"created session [bold]{s.id}[/]")
    return 0


@session.command("show")
@click.argument("session_id")
@click.option("--limit", default=50, show_default=True)
@runtime_command()
async def session_show(ctx: CLIContext, rt: Any, session_id: str, limit: int) -> int:
    s = rt.sessions.resolve(session_id, project_id=rt.require_project().id)
    messages = rt.sessions.messages(s.id, limit=limit)
    tasks = rt.tasks.list(session_id=s.id, limit=100)

    def human(c: Any) -> None:
        c.print(
            f"[bold]{s.title}[/] {s.id} · {s.status}"
            + (f" · forked from {s.parent_session_id}" if s.parent_session_id else "")
        )
        for t in reversed(tasks):
            c.print(f"  task {t.id} {styled_status(t.status.value)} {one_line(t.title, 70)}")
        for m in messages:
            c.rule(f"{m.role} · {ago(m.created_at)}", style="dim")
            ctx.output.markdown(m.content)

    ctx.output.data(
        {
            "session": s.__dict__,
            "messages": [m.__dict__ for m in messages],
            "tasks": [t.to_dict() for t in tasks],
        },
        human,
    )
    return 0


@session.command("fork")
@click.argument("session_id")
@click.option("--title", default=None)
@runtime_command()
async def session_fork(ctx: CLIContext, rt: Any, session_id: str, title: str | None) -> int:
    """Fork a session to explore an alternative without touching the main working state.

    Tasks in a forked session keep verified changes in their isolated workspaces (no auto-apply)
    so approaches can be compared with `core task diff` and applied deliberately.
    """
    project = rt.require_project()
    parent = rt.sessions.resolve(session_id, project_id=project.id)
    ws = rt.workspaces.canonical(project)
    checkpoint_ref = None
    try:
        checkpoint_ref = await rt.workspaces.fingerprint(ws)
    except Exception as exc:  # noqa: BLE001 - checkpointing is best-effort for the fork record
        ctx.output.err.print(f"[yellow]could not snapshot the working tree: {exc}[/]")
    from coremain.util.ids import new_id

    chk_id = new_id("chk")
    rt.db.execute(
        "INSERT INTO checkpoints(id, project_id, session_id, kind, label, tree_ref, created_at) VALUES (?,?,?,?,?,?,?)",
        (chk_id, project.id, parent.id, "session_fork", f"fork of {parent.id}", checkpoint_ref, time.time()),
    )
    child = rt.sessions.create(
        project.id, title or f"Fork of {parent.title}", parent_session_id=parent.id, fork_checkpoint_id=chk_id
    )
    for m in rt.sessions.messages(parent.id, limit=6):
        rt.sessions.add_message(
            child.id, "system", f"[from parent session, {m.role}] {one_line(m.content, 1500)}"
        )
    ctx.output.data(
        child.__dict__,
        f"forked [bold]{child.id}[/] from {parent.id} (checkpoint {chk_id}); tasks here will not auto-apply",
    )
    return 0


@session.command("archive")
@click.argument("session_id")
@runtime_command()
async def session_archive(ctx: CLIContext, rt: Any, session_id: str) -> int:
    s = rt.sessions.resolve(session_id, project_id=rt.require_project().id)
    rt.sessions.archive(s.id)
    ctx.output.data({"archived": s.id}, f"archived {s.id}")
    return 0


@session.command("search")
@click.argument("query", nargs=-1, required=True)
@runtime_command()
async def session_search(ctx: CLIContext, rt: Any, query: tuple[str, ...]) -> int:
    hits = rt.sessions.search(rt.require_project().id, " ".join(query))
    rows = [
        {
            "session_id": m.session_id,
            "message_id": m.id,
            "role": m.role,
            "snippet": snip,
            "task_id": m.task_id,
        }
        for m, snip in hits
    ]
    ctx.output.data(
        rows,
        lambda c: ctx.output.table(
            ["session", "role", "snippet"], [[r["session_id"], r["role"], r["snippet"]] for r in rows]
        ),
    )
    return 0


# ============================================================================ tasks
@click.group()
def task() -> None:
    """Inspect and control tasks."""


@task.command("list")
@click.option("--status", "statuses", multiple=True, type=click.Choice([s.value for s in TaskStatus]))
@click.option("--session", "session_id")
@click.option("--limit", default=30, show_default=True)
@runtime_command()
async def task_list(
    ctx: CLIContext, rt: Any, statuses: tuple[str, ...], session_id: str | None, limit: int
) -> int:
    project = rt.require_project()
    tasks = rt.tasks.list(
        project_id=project.id,
        session_id=session_id,
        statuses=[TaskStatus(s) for s in statuses] or None,
        limit=limit,
    )
    ctx.output.data(
        [t.to_dict() for t in tasks],
        lambda c: ctx.output.table(
            ["id", "status", "mode", "kind", "title", "updated", "reason"],
            [
                [
                    t.id,
                    styled_status(t.status.value),
                    t.mode or "",
                    t.kind,
                    one_line(t.title, 50),
                    ago(t.updated_at),
                    one_line(t.status_reason or "", 50),
                ]
                for t in tasks
            ],
        ),
    )
    return 0


@task.command("show")
@click.argument("task_id")
@runtime_command()
async def task_show(ctx: CLIContext, rt: Any, task_id: str) -> int:
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    attempts = rt.tasks.attempts(t.id)
    evidence = rt.evidence.for_task(t.id)
    findings = rt.reviews.open_findings(t.id)
    ws = rt.workspaces.get(t.workspace_id) if t.workspace_id else None
    data = {
        "task": t.to_dict(),
        "attempts": [a.to_dict() for a in attempts],
        "workspace": ws.to_dict() if ws else None,
        "evidence": [e.to_dict() for e in evidence],
        "open_findings": findings,
        "children": [c.to_dict() for c in rt.tasks.list(parent_task_id=t.id)],
    }

    def human(c: Any) -> None:
        c.print(
            f"[bold]{t.title}[/]\n{t.id} · {styled_status(t.status.value)} · mode {t.mode} · kind {t.kind} · workflow {t.decision.get('workflow')}"
        )
        if t.status_reason:
            c.print(
                f"reason: {t.status_reason}" + (f" (blocked on {t.block_reason})" if t.block_reason else "")
            )
        if t.decision.get("reasons"):
            c.print("[dim]mode selection: " + "; ".join(t.decision["reasons"]) + "[/]")
        reqs = t.contract.get("requirements", [])
        if reqs:
            c.print("[bold]Completion requirements[/]")
            for r in reqs:
                c.print(
                    f"  - {r.get('description') or r['kind']}"
                    + ("" if r.get("required", True) else " [dim](optional)[/]")
                )
        if ws:
            c.print(
                f"[bold]Workspace[/] {ws.kind} {ws.path} ({ws.status})"
                + (f" branch {ws.branch}" if ws.branch else "")
            )
        for a in attempts:
            c.print(
                f"  attempt {a.number} {a.status.value} {ago(a.started_at)}"
                + (f" — {a.error_class}: {one_line(a.error_message or '', 80)}" if a.error_class else "")
                + (f" [dim](resumed from {a.resumed_from})[/]" if a.resumed_from else "")
            )
        if evidence:
            c.print("[bold]Evidence[/]")
            for e in evidence[-20:]:
                c.print(
                    f"  {e.kind:<18} {styled_status('completed' if e.status == 'pass' else 'failed' if e.status == 'fail' else 'unknown')[:-3]} "
                    f"{e.status:<12} [dim]{e.trust:<9}[/] {one_line(e.summary, 80)}"
                )
        for f in findings:
            c.print(
                f"  [yellow]{f['severity']}[/] {f['title']} {f.get('file') or ''}{':' + str(f['line']) if f.get('line') else ''}"
            )
        if t.result_summary:
            c.print(f"[bold]Result[/] {one_line(t.result_summary, 400)}")

    ctx.output.data(data, human)
    return 0


@task.command("events")
@click.argument("task_id")
@click.option("--kind", "kinds", multiple=True, help="Filter by event kind.")
@runtime_command()
async def task_events(ctx: CLIContext, rt: Any, task_id: str, kinds: tuple[str, ...]) -> int:
    """Timeline of durable events for a task (trace)."""
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    events = rt.events.for_task(t.id, kinds=kinds or None)
    ctx.output.data(
        [e.to_dict() for e in events],
        lambda c: [
            c.print(
                f"[dim]{time.strftime('%H:%M:%S', time.localtime(e.ts))}[/] {e.kind:<24} {one_line(dumps(e.data), 140)}"
            )
            for e in events
        ],
    )
    return 0


@task.command("diff")
@click.argument("task_id")
@click.option("--stat", is_flag=True)
@runtime_command()
async def task_diff(ctx: CLIContext, rt: Any, task_id: str, stat: bool) -> int:
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    if not t.workspace_id:
        raise NotFoundError(f"task {t.id} has no workspace")
    ws = rt.workspaces.get(t.workspace_id)
    if not ws.path.exists():
        raise NotFoundError(f"workspace {ws.id} no longer exists on disk ({ws.status})")
    diff = await rt.workspaces.diff(ws)
    payload = {
        "files": [f.__dict__ for f in diff.files],
        "stats": diff.stats,
        "diff_hash": diff.diff_hash,
        "patch": None if stat else diff.patch,
    }

    def human(c: Any) -> None:
        if stat or not diff.patch:
            for f in diff.files:
                c.print(f"{f.status} {f.path}")
            c.print(f"[dim]{diff.stats}[/]")
        else:
            from rich.syntax import Syntax

            c.print(Syntax(diff.patch, "diff", theme="ansi_dark", word_wrap=False))

    ctx.output.data(payload, human)
    return 0


@task.command("apply")
@click.argument("task_id")
@click.option(
    "--force",
    is_flag=True,
    help="Apply even if the task is not completed (you take responsibility for unverified changes).",
)
@click.option("--dry-run", is_flag=True)
@runtime_command()
async def task_apply(ctx: CLIContext, rt: Any, task_id: str, force: bool, dry_run: bool) -> int:
    """Apply a task's isolated workspace to the working tree (3-way, all-or-nothing)."""
    project = rt.require_project()
    t = rt.tasks.resolve(task_id, project_id=project.id)
    if not t.workspace_id:
        raise NotFoundError(f"task {t.id} has no workspace")
    ws = rt.workspaces.get(t.workspace_id)
    if t.status != TaskStatus.COMPLETED and not force:
        raise ConflictError(
            f"task is {t.status.value}; its changes are not verified",
            hint="Pass --force to apply unverified changes anyway.",
        )
    if ws.status == "applied":
        ctx.output.data({"applied": False, "reason": "already applied"}, "already applied")
        return 0
    if dry_run:
        res = await rt.workspaces.apply(ws, Path(project.root_path), dry_run=True)
        ctx.output.data(res.to_dict(), lambda c: c.print(res.to_dict()))
        return 0 if res.ok else 7
    res = await rt.workspaces.apply_to_canonical(ws, project)
    rt.events.emit(
        "task.applied_manually",
        project_id=project.id,
        task_id=t.id,
        actor="user",
        data={**res.to_dict(), "forced": force},
    )
    ctx.output.data(res.to_dict(), f"applied {len(res.applied)} file(s), merged {len(res.merged)}")
    return 0


@task.command("discard")
@click.argument("task_id")
@click.option("--yes", is_flag=True)
@runtime_command()
async def task_discard(ctx: CLIContext, rt: Any, task_id: str, yes: bool) -> int:
    """Delete a task's isolated workspace (never for running tasks)."""
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    if t.status in {TaskStatus.RUNNING, TaskStatus.VERIFYING, TaskStatus.REVIEWING, TaskStatus.QUEUED}:
        raise ConflictError(f"task {t.id} is {t.status.value}; cancel it first")
    if not t.workspace_id:
        raise NotFoundError("task has no workspace")
    ws = rt.workspaces.get(t.workspace_id)
    if not ws.isolated:
        raise UsageError("only isolated workspaces can be discarded")
    if not yes and not ctx.output.json_mode and not click.confirm(f"Delete {ws.path}?", default=False):
        return 8
    await rt.workspaces.remove(ws, reason="discarded by user")
    ctx.output.data({"discarded": ws.id}, f"discarded workspace {ws.id}")
    return 0


@task.command("cancel")
@click.argument("task_id")
@click.option("--reason", default="cancelled by user")
@runtime_command()
async def task_cancel(ctx: CLIContext, rt: Any, task_id: str, reason: str) -> int:
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    t = rt.cancel_task(t.id, reason=reason)
    ctx.output.data(
        t.to_dict(),
        f"{t.id}: {styled_status(t.status.value)}"
        + (" (cancellation requested; the worker will stop)" if t.status.value != "cancelled" else ""),
    )
    return 0


@task.command("pause")
@click.argument("task_id")
@runtime_command()
async def task_pause(ctx: CLIContext, rt: Any, task_id: str) -> int:
    t = rt.tasks.request_pause(rt.tasks.resolve(task_id, project_id=rt.require_project().id).id)
    ctx.output.data(
        t.to_dict(),
        f"{t.id}: pause requested (takes effect at the next node boundary)"
        if t.status != TaskStatus.PAUSED
        else f"{t.id}: paused",
    )
    return 0


@task.command("resume")
@click.argument("task_id")
@click.option("--run", "run_now", is_flag=True, help="Run it now in the foreground.")
@click.option("--note", default=None, help="Note recorded with the resume.")
@runtime_command()
async def task_resume(ctx: CLIContext, rt: Any, task_id: str, run_now: bool, note: str | None) -> int:
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    if t.status != TaskStatus.QUEUED:
        t = rt.tasks.resume(t.id, note=note)
    if not run_now:
        ctx.output.data(t.to_dict(), f"{t.id} queued; run with `core task resume {t.id} --run`")
        return 0
    from coremain.cli.progress import ProgressPrinter, event_filter

    await rt.start()
    sub = rt.bus.subscribe(event_filter({t.id}))
    printer = ProgressPrinter(ctx.output, jsonl=False, verbose=ctx.output.verbose)
    import asyncio

    consumer = asyncio.create_task(printer.consume(sub)) if not ctx.output.json_mode else None
    final = await rt.run_task(t.id)
    if consumer:
        consumer.cancel()
    sub.close()
    ctx.output.data(
        final.to_dict(),
        lambda c: c.print(f"{final.id}: {styled_status(final.status.value)} — {final.status_reason or ''}"),
    )
    return int(task_exit_code(final.status))


@task.command("retry")
@click.argument("task_id")
@runtime_command()
async def task_retry(ctx: CLIContext, rt: Any, task_id: str) -> int:
    """Re-run a failed/incomplete/blocked task from its last checkpoint."""
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    t = rt.tasks.resume(t.id, note="retry requested by user")
    final = await rt.run_task(t.id)
    ctx.output.data(
        final.to_dict(),
        lambda c: c.print(f"{final.id}: {styled_status(final.status.value)} — {final.status_reason or ''}"),
    )
    return int(task_exit_code(final.status))


@task.command("answer")
@click.argument("task_id")
@click.argument("text", nargs=-1, required=True)
@runtime_command()
async def task_answer(ctx: CLIContext, rt: Any, task_id: str, text: tuple[str, ...]) -> int:
    """Answer a task's pending question (then resume it)."""
    t = rt.answer(rt.tasks.resolve(task_id, project_id=rt.require_project().id).id, " ".join(text))
    ctx.output.data(t.to_dict(), f"{t.id}: answer recorded; status {styled_status(t.status.value)}")
    return 0


@task.command("evidence")
@click.argument("task_id")
@runtime_command()
async def task_evidence(ctx: CLIContext, rt: Any, task_id: str) -> int:
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    ev = rt.evidence.for_task(t.id)
    ctx.output.data(
        [e.to_dict() for e in ev],
        lambda c: ctx.output.table(
            ["kind", "status", "trust", "attempt", "fingerprint", "summary"],
            [
                [
                    e.kind,
                    e.status,
                    e.trust,
                    (e.attempt_id or "")[-8:],
                    (e.fingerprint or "")[:10],
                    one_line(e.summary, 70),
                ]
                for e in ev
            ],
        ),
    )
    return 0


@task.command("findings")
@click.argument("task_id")
@click.option("--all", "all_findings", is_flag=True, help="Include resolved/dismissed findings.")
@runtime_command()
async def task_findings(ctx: CLIContext, rt: Any, task_id: str, all_findings: bool) -> int:
    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    rows = (
        [
            dict(r)
            for r in rt.db.query("SELECT * FROM findings WHERE task_id = ? ORDER BY created_at", (t.id,))
        ]
        if all_findings
        else rt.reviews.open_findings(t.id)
    )
    ctx.output.data(
        rows,
        lambda c: ctx.output.table(
            ["severity", "category", "title", "location", "status"],
            [
                [
                    r["severity"],
                    r["category"],
                    one_line(r["title"], 60),
                    f"{r.get('file') or ''}:{r.get('line') or ''}",
                    r["status"],
                ]
                for r in rows
            ],
        ),
    )
    return 0


@task.command("context")
@click.argument("task_id")
@click.option("--show", "snapshot_id", default=None, help="Print the compiled content of a snapshot.")
@runtime_command()
async def task_context(ctx: CLIContext, rt: Any, task_id: str, snapshot_id: str | None) -> int:
    """Inspect what context was compiled for each model call (manifests and content)."""
    from coremain.util.jsonutil import loads

    t = rt.tasks.resolve(task_id, project_id=rt.require_project().id)
    if snapshot_id:
        row = rt.db.one(
            "SELECT * FROM context_snapshots WHERE task_id = ? AND (id = ? OR id LIKE ?)",
            (t.id, snapshot_id, f"%{snapshot_id}"),
        )
        if row is None:
            raise NotFoundError(f"snapshot {snapshot_id} not found for task {t.id}")
        content = (
            rt.artifacts.read_text(row["content_artifact_id"], project_id=t.project_id)
            if row["content_artifact_id"]
            else ""
        )
        ctx.output.data(
            {"manifest": loads(row["manifest_json"]), "content": content},
            lambda c: c.print(content, markup=False),
        )
        return 0
    rows = rt.db.query("SELECT * FROM context_snapshots WHERE task_id = ? ORDER BY created_at", (t.id,))
    data = [
        {
            "id": r["id"],
            "stage": r["stage"],
            "strategy": f"{r['strategy']}@{r['strategy_version']}",
            "budget": r["budget_tokens"],
            "used": r["used_tokens"],
            "included": [f"{i['kind']}:{i['source']}" for i in loads(r["manifest_json"])["included"]],
        }
        for r in rows
    ]
    ctx.output.data(
        data,
        lambda c: ctx.output.table(
            ["snapshot", "stage", "strategy", "used/budget", "included"],
            [
                [
                    d["id"],
                    d["stage"],
                    d["strategy"],
                    f"{d['used']}/{d['budget']}",
                    one_line(", ".join(d["included"]), 90),
                ]
                for d in data
            ],
        ),
    )
    return 0


# ============================================================================ approvals
@click.group()
def approvals() -> None:
    """Pending human approvals and questions."""


@approvals.command("list")
@runtime_command()
async def approvals_list(ctx: CLIContext, rt: Any) -> int:
    rows = [a.to_dict() for a in rt.approvals.pending(project_id=rt.require_project().id)]
    ctx.output.data(
        rows,
        lambda c: ctx.output.table(
            ["id", "task", "capability", "summary", "age"],
            [
                [r["id"], r["task_id"], r["capability"], one_line(r["summary"], 70), ago(r["created_at"])]
                for r in rows
            ],
        ),
    )
    return 0


@approvals.command("approve")
@click.argument("approval_id")
@click.option(
    "--scope", type=click.Choice(["once", "task", "session", "project"]), default="once", show_default=True
)
@runtime_command()
async def approvals_approve(ctx: CLIContext, rt: Any, approval_id: str, scope: str) -> int:
    a = rt.decide_approval(rt.approvals.resolve(approval_id).id, True, scope=scope)
    ctx.output.data(
        a.to_dict(),
        f"approved {a.id} ({scope}); resume with `core task resume {a.task_id} --run`"
        if a.task_id
        else f"approved {a.id}",
    )
    return 0


@approvals.command("reject")
@click.argument("approval_id")
@click.option("--reason", default="rejected by user")
@click.option(
    "--scope", type=click.Choice(["once", "task", "session", "project"]), default="once", show_default=True
)
@runtime_command()
async def approvals_reject(ctx: CLIContext, rt: Any, approval_id: str, reason: str, scope: str) -> int:
    a = rt.decide_approval(rt.approvals.resolve(approval_id).id, False, scope=scope, reason=reason)
    ctx.output.data(a.to_dict(), f"rejected {a.id}")
    return 0
