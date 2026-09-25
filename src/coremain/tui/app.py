"""The Core Main terminal UI (Textual).

The TUI is a client of the same ``CoreRuntime`` the CLI uses, running on the same event loop:
tasks execute in-process, approvals and agent questions become modal dialogs, and the screen is
driven by runtime events (durable and streaming). Nothing here owns domain state; every panel is
re-read from the database, so tasks started elsewhere (e.g. `core run` in another terminal)
show up too.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from typing import TYPE_CHECKING, Any

from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Input, ListView, RichLog, Static, TabbedContent, TabPane

from coremain.domain.states import ATTENTION, TERMINAL, TaskStatus
from coremain.errors import CoreError
from coremain.tools.base import ApprovalDecision
from coremain.tui.commands import MODES, CoreCommands
from coremain.tui.screens import ApprovalScreen, HelpScreen, QuestionScreen
from coremain.tui.widgets import Conversation, SessionItem, SessionList, StatusBar, TaskBlock
from coremain.ui.describe import describe_event, esc, styled_status
from coremain.util.text import one_line

if TYPE_CHECKING:
    from coremain.cli.common import CLIContext
    from coremain.events import Event, Subscription
    from coremain.runtime.app import CoreRuntime

DONE = {s.value for s in TERMINAL | ATTENTION}


class CoreApp(App[int]):
    CSS_PATH = "app.tcss"
    TITLE = "Core Main"
    COMMANDS = App.COMMANDS | {CoreCommands}
    BINDINGS = [
        Binding("ctrl+n", "new_session", "New session"),
        Binding("ctrl+k", "cancel_task", "Cancel task"),
        Binding("ctrl+b", "toggle_sessions", "Sessions"),
        Binding("f1", "help", "Help"),
        Binding("f2", "tab('tab-task')", "Task", show=False),
        Binding("f3", "tab('tab-evidence')", "Evidence", show=False),
        Binding("f4", "tab('tab-diff')", "Diff", show=False),
        Binding("f5", "tab('tab-activity')", "Activity", show=False),
        Binding("f6", "tab('tab-approvals')", "Approvals", show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        ctx: CLIContext | None = None,
        *,
        session_id: str | None = None,
        runtime: CoreRuntime | None = None,
    ):
        super().__init__()
        self.ctx = ctx
        self.session_arg = session_id
        self.rt: CoreRuntime | None = runtime
        self._owns_runtime = runtime is None
        self.session_id: str | None = None
        self.current_task_id: str | None = None
        self.mode_override: str | None = None
        self.model_override: str | None = None
        self.routes: dict[str, dict[str, str]] = {}
        self._sub: Subscription | None = None
        self._evidence_count = -1
        self._quit_armed = 0.0

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield StatusBar(id="status")
        with Horizontal(id="body"):
            yield SessionList(id="sessions")
            with Vertical(id="main"):
                yield Conversation(id="conversation")
                with Vertical(id="prompt"):
                    yield Input(
                        placeholder="Describe a task or ask a question…  (/help for commands)", id="input"
                    )
                    yield Static("Enter to send · ctrl+p commands · f1 help", id="prompt-hint")
            with TabbedContent(id="panel", initial="tab-task"):
                with TabPane("Task", id="tab-task"), VerticalScroll():
                    yield Static("No task yet.", id="task-details")
                with TabPane("Evidence", id="tab-evidence"):
                    yield DataTable(id="evidence", zebra_stripes=True, cursor_type="row")
                with TabPane("Diff", id="tab-diff"), VerticalScroll():
                    yield Static("No changes to show.", id="diff-view")
                with TabPane("Activity", id="tab-activity"):
                    yield RichLog(id="activity", markup=False, wrap=True, max_lines=2000)
                with TabPane("Approvals", id="tab-approvals"):
                    yield DataTable(id="approvals", cursor_type="row")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            if self.rt is None:
                assert self.ctx is not None
                self.rt = self.ctx.open_runtime(mode="tui")
            await self.rt.start()
        except CoreError as exc:
            self.exit(return_code=int(exc.exit_code), message=f"core: {exc.message}")
            return
        rt = self.rt
        rt.approval_handler = self._approval
        rt.input_handler = self._question
        project = rt.require_project()
        self.query_one("#evidence", DataTable).add_columns("kind", "status", "trust", "summary")
        self.query_one("#approvals", DataTable).add_columns("id", "capability", "summary")
        session = rt.ensure_session(self.session_arg)
        await self._load_sessions()
        await self.open_session(session.id)
        self._sub = rt.bus.subscribe(lambda e: e.project_id in (None, project.id))
        self.run_worker(self._pump(), name="events", group="events", exclusive=True)
        self.set_interval(1.0, self.refresh_panels)
        self.query_one("#input", Input).focus()

    async def on_unmount(self) -> None:
        if self._sub is not None:
            self._sub.close()
        if self.rt is not None and self._owns_runtime:
            with contextlib.suppress(Exception):
                await self.rt.close()

    # ---------------------------------------------------------------- sessions
    async def _load_sessions(self) -> None:
        assert self.rt is not None
        view = self.query_one("#sessions", SessionList)
        await view.clear()
        for session in self.rt.sessions.list(self.rt.require_project().id, limit=50):
            await view.append(SessionItem(session))

    async def open_session(self, session_id: str) -> None:
        assert self.rt is not None
        rt = self.rt
        self.session_id = session_id
        conv = self.query_one("#conversation", Conversation)
        await conv.clear()
        messages = rt.sessions.messages(session_id, limit=400)
        if not messages:
            await conv.add_note(self._welcome())
        for m in messages:
            if m.role == "user":
                await conv.add_user(m.content)
            elif m.role == "assistant":
                await conv.add_markdown(m.content)
        running = [
            t
            for t in rt.tasks.list(project_id=rt.require_project().id, session_id=session_id, limit=20)
            if t.status.value not in DONE
        ]
        for t in running:
            await conv.add_task(t.id, t.title, t.status.value)
            self.current_task_id = t.id
        if not running:
            latest = rt.tasks.list(project_id=rt.require_project().id, session_id=session_id, limit=1)
            self.current_task_id = latest[0].id if latest else None
        self._evidence_count = -1
        self.refresh_panels()

    def _welcome(self) -> str:
        assert self.rt is not None
        models = self.rt.registry.models()
        lines = [
            "[b]Welcome to Core Main.[/b] Describe what you want done, or ask a question about this repository."
        ]
        if models:
            lines.append(
                f"[dim]Models: {esc(', '.join(m.key for m in models[:6]))}. Press f1 for commands.[/]"
            )
        else:
            lines.append(
                "[yellow]No models are configured yet.[/] Add a provider and model, e.g. "
                "`core providers login openai` then `core models add …` (see README), or run `core doctor`."
            )
        return "\n".join(lines)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, SessionItem) and event.item.session_id != self.session_id:
            await self.open_session(event.item.session_id)
            self.query_one("#input", Input).focus()

    # ------------------------------------------------------------------ input
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            await self.run_slash(text)
        else:
            await self.submit(text)

    async def submit(self, text: str, *, mode: str | None = None) -> None:
        assert self.rt is not None
        conv = self.query_one("#conversation", Conversation)
        await conv.add_user(text)
        try:
            task = await self.rt.submit(
                text, session_id=self.session_id, mode=mode or self.mode_override, model=self.model_override
            )
        except CoreError as exc:
            await conv.add_note(
                f"[red]{esc(exc.message)}[/]"
                + (f"\n[dim]{esc(exc.hint)}[/]" if getattr(exc, "hint", None) else ""),
                classes="note error",
            )
            return
        self.current_task_id = task.id
        block = await conv.add_task(task.id, task.title, task.status.value)
        cls = (task.decision or {}).get("classification") or {}
        block.add_line(
            f"[bold]▸ mode[/] {esc(task.mode)} [dim]({esc('; '.join((task.decision or {}).get('reasons', [])[:3]))})[/]"
        )
        if cls.get("risk"):
            block.add_line(
                f"[dim]  risk {esc(cls['risk'])} · workflow {esc((task.decision or {}).get('workflow'))}[/]"
            )
        self.rt.start_task(task.id)
        self._evidence_count = -1
        self.refresh_panels()
        if len(conv.blocks) == 1:
            await self._load_sessions()  # the first request titles the session

    async def run_slash(self, text: str) -> None:
        assert self.rt is not None
        rt = self.rt
        conv = self.query_one("#conversation", Conversation)
        name, _, arg = text[1:].partition(" ")
        arg = arg.strip()
        try:
            if name in {"help", "h", "?"}:
                self.action_help()
            elif name == "ask":
                if not arg:
                    await conv.add_note("usage: /ask <question>")
                else:
                    await self.submit(arg, mode="answer")
            elif name == "mode":
                if arg not in {"", "auto", *MODES}:
                    await conv.add_note(
                        f"[yellow]unknown mode {esc(arg)}; choose one of {', '.join(MODES)} or auto[/]"
                    )
                else:
                    self.mode_override = None if arg in {"", "auto"} else arg
                    await conv.add_note(f"mode for new requests: {self.mode_override or 'automatic'}")
            elif name == "model":
                if arg in {"", "auto"}:
                    self.model_override = None
                else:
                    rt.registry.model(arg)
                    self.model_override = arg
                await conv.add_note(f"model for new requests: {self.model_override or 'automatic (router)'}")
            elif name == "new":
                session = rt.sessions.create(rt.require_project().id, arg or "New session")
                await self._load_sessions()
                await self.open_session(session.id)
            elif name == "cancel":
                self.action_cancel_task()
            elif name == "resume":
                task = self._require_task()
                if task.status.value != "queued":
                    rt.tasks.resume(task.id, note="resumed from the TUI")
                block = conv.blocks.get(task.id) or await conv.add_task(task.id, task.title, "queued")
                block.reopen()
                rt.start_task(task.id)
            elif name == "diff":
                self.action_tab("tab-diff")
                await self.show_diff()
            elif name == "apply":
                await self.apply_current()
            elif name in {"approve", "deny"}:
                pending = rt.approvals.pending(project_id=rt.require_project().id)
                if not pending:
                    await conv.add_note("no pending approvals")
                else:
                    rt.decide_approval(pending[0].id, name == "approve", reason="decided in the TUI")
                    if pending[0].task_id and rt.tasks.get(pending[0].task_id).status == TaskStatus.QUEUED:
                        rt.start_task(pending[0].task_id)
                    await conv.add_note(
                        f"{'approved' if name == 'approve' else 'denied'}: {esc(pending[0].summary)}"
                    )
            elif name == "tasks":
                tasks = rt.tasks.list(
                    project_id=rt.require_project().id, session_id=self.session_id, limit=15
                )
                await conv.add_note(
                    "\n".join(
                        f"{styled_status(t.status.value)} {esc(one_line(t.title, 70))} [dim]{t.id}[/]"
                        for t in tasks
                    )
                    or "no tasks in this session"
                )
            elif name in {"quit", "exit", "q"}:
                await self.action_quit()
            else:
                await conv.add_note(f"[yellow]unknown command /{esc(name)} — /help lists commands[/]")
        except CoreError as exc:
            await conv.add_note(f"[red]{esc(exc.message)}[/]", classes="note error")

    def _require_task(self) -> Any:
        assert self.rt is not None
        if not self.current_task_id:
            raise CoreError("there is no current task")
        return self.rt.tasks.get(self.current_task_id)

    async def apply_current(self) -> None:
        assert self.rt is not None
        rt = self.rt
        conv = self.query_one("#conversation", Conversation)
        task = self._require_task()
        if task.status != TaskStatus.COMPLETED:
            await conv.add_note(
                f"[yellow]task is {task.status.value}; only verified (completed) work is applied from the TUI. "
                f"Use `core task apply {task.id} --force` to override.[/]"
            )
            return
        if not task.workspace_id:
            await conv.add_note("the task has no workspace")
            return
        ws = rt.workspaces.get(task.workspace_id)
        if ws.status == "applied" or not ws.isolated:
            await conv.add_note("already applied")
            return
        res = await rt.workspaces.apply_to_canonical(ws, rt.require_project())
        rt.events.emit(
            "task.applied_manually",
            project_id=task.project_id,
            task_id=task.id,
            actor="user",
            data=res.to_dict(),
        )
        await conv.add_note(f"[green]applied {len(res.applied)} file(s), merged {len(res.merged)}[/]")

    # ------------------------------------------------------------- approvals
    async def _modal(self, screen: Any) -> Any:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        def done(result: Any) -> None:
            if not future.done():
                future.set_result(result)

        self.push_screen(screen, callback=done)
        self.bell()
        try:
            return await future
        except asyncio.CancelledError:
            if self.screen is screen:
                self.pop_screen()
            raise

    async def _approval(self, approval: Any) -> ApprovalDecision:
        return await self._modal(ApprovalScreen(approval))

    async def _question(self, question: str, context: str | None) -> str:
        return await self._modal(QuestionScreen(question, context))

    # ---------------------------------------------------------------- events
    async def _pump(self) -> None:
        assert self._sub is not None
        async for ev in self._sub:
            try:
                await self.handle_event(ev)
            except Exception as exc:  # noqa: BLE001 - a rendering bug must not stop event delivery
                self.log.error(f"event {ev.kind}: {exc}")

    async def handle_event(self, ev: Event) -> None:
        conv = self.query_one("#conversation", Conversation)
        block: TaskBlock | None = conv.blocks.get(ev.task_id or "")
        if ev.kind == "model.delta":
            if block is not None:
                await block.stream(str(ev.data.get("text", "")))
            return
        if ev.kind == "model.started":
            if block is not None:
                await block.reset_live()
            return
        if ev.kind == "route.decision" and ev.task_id:
            self.routes.setdefault(ev.task_id, {})[str(ev.data.get("role"))] = str(ev.data.get("model"))
        if ev.kind == "task.created" and ev.data.get("parent") in conv.blocks:
            parent = conv.blocks[ev.data["parent"]]
            parent.add_line(f"[dim]  ↳ subtask {esc(one_line(str(ev.data.get('title', '')), 70))}[/]")
        line = describe_event(ev)
        if line is None:
            return
        stamp = time.strftime("%H:%M:%S", time.localtime(ev.ts))
        self.query_one("#activity", RichLog).write(
            Text.from_markup(f"[dim]{stamp} {(ev.task_id or '')[-6:]}[/] {line}")
        )
        if block is None:
            return
        if ev.kind == "task.transition":
            status = str(ev.data.get("to"))
            block.set_status(status, str(ev.data.get("reason") or ""))
            if status in DONE:
                await block.show_report(
                    self._report(ev.task_id or "") or f"Task {status}: {ev.data.get('reason')}"
                )
                if status in {"awaiting_approval", "needs_input"}:
                    block.reopen()
            elif block.done:
                block.reopen()
        else:
            block.add_line(line)
        if ev.kind == "approval.requested":
            self.refresh_panels()

    def _report(self, task_id: str) -> str | None:
        assert self.rt is not None
        task = self.rt.tasks.get(task_id)
        if not task.session_id:
            return None
        msgs = [
            m
            for m in self.rt.sessions.messages(task.session_id, limit=200)
            if m.task_id == task_id and m.role == "assistant"
        ]
        return msgs[-1].content if msgs else None

    # ---------------------------------------------------------------- panels
    def refresh_panels(self) -> None:
        if self.rt is None:
            return
        rt = self.rt
        project = rt.require_project()
        session = rt.sessions.get(self.session_id) if self.session_id else None
        self.query_one("#status", StatusBar).render_status(
            project=project.name,
            session=session.title if session else "-",
            mode=self.mode_override,
            model=self.model_override,
            running=len(rt._running),
            offline=rt.config.permissions.network.offline,
            models=len(rt.registry.models()),
        )
        approvals = rt.approvals.pending(project_id=project.id)
        table = self.query_one("#approvals", DataTable)
        table.clear()
        for a in approvals:
            table.add_row(a.id[-8:], a.capability, one_line(a.summary, 60), key=a.id)
        if not self.current_task_id:
            return
        task = rt.tasks.get(self.current_task_id)
        decision = task.decision or {}
        reqs = task.contract.get("requirements") or []
        lines = [
            f"[b]{esc(task.title)}[/b]",
            f"{styled_status(task.status.value)}  [dim]{esc(task.status_reason or '')}[/]",
            "",
            f"[dim]task[/]      {task.id}",
            f"[dim]kind[/]      {esc(task.kind)}   [dim]mode[/] {esc(task.mode)}   [dim]workflow[/] {esc(decision.get('workflow'))}",
            f"[dim]why[/]       {esc('; '.join(decision.get('reasons', [])[:4]))}",
            f"[dim]attempts[/]  {task.attempt_count}"
            + (f"  [dim]recovered[/] {task.recovered_count}" if task.recovered_count else ""),
            f"[dim]evidence[/]  {esc(task.evidence_level or '-')}",
        ]
        if reqs:
            lines.append(
                "[dim]requires[/]  "
                + esc(", ".join(r.get("description") or r["kind"] for r in reqs if r.get("required", True)))
            )
        for role, model in sorted(self.routes.get(task.id, {}).items()):
            lines.append(f"[dim]{esc(role):<10}[/]{esc(model)}")
        if task.workspace_id:
            with contextlib.suppress(CoreError):
                ws = rt.workspaces.get(task.workspace_id)
                lines.append(f"[dim]workspace[/] {esc(ws.kind)} {esc(ws.status)} [dim]{esc(ws.path)}[/]")
        if task.result_summary:
            lines += ["", esc(one_line(task.result_summary, 400))]
        self.query_one("#task-details", Static).update("\n".join(lines))
        evidence = rt.evidence.for_task(task.id)
        if len(evidence) != self._evidence_count:
            self._evidence_count = len(evidence)
            ev_table = self.query_one("#evidence", DataTable)
            ev_table.clear()
            for e in evidence:
                mark = {"pass": "[green]pass[/]", "fail": "[red]fail[/]"}.get(e.status, e.status)
                ev_table.add_row(e.kind, Text.from_markup(mark), e.trust, one_line(e.summary, 70))

    async def show_diff(self) -> None:
        assert self.rt is not None
        view = self.query_one("#diff-view", Static)
        if not self.current_task_id:
            view.update("No task selected.")
            return
        task = self.rt.tasks.get(self.current_task_id)
        if not task.workspace_id:
            view.update("This task has no workspace.")
            return
        ws = self.rt.workspaces.get(task.workspace_id)
        if not ws.path.exists():
            view.update(f"Workspace {ws.status}; its files are no longer on disk.")
            return
        diff = await self.rt.workspaces.diff(ws)
        if not diff.patch.strip():
            view.update("No changes.")
            return
        view.update(Syntax(diff.patch, "diff", theme="ansi_dark", word_wrap=False))

    async def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "tab-diff":
            await self.show_diff()

    # ---------------------------------------------------------------- actions
    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_tab(self, tab: str) -> None:
        self.query_one("#panel", TabbedContent).active = tab

    def action_toggle_sessions(self) -> None:
        self.query_one("#sessions").toggle_class("hidden")

    async def action_new_session(self) -> None:
        await self.run_slash("/new")

    def action_cancel_task(self) -> None:
        if self.rt is None or not self.current_task_id:
            self.notify("no current task", severity="warning")
            return
        task = self.rt.tasks.get(self.current_task_id)
        if task.status.value in {s.value for s in TERMINAL}:
            self.notify(f"task already {task.status.value}")
            return
        self.rt.cancel_task(task.id, reason="cancelled in the TUI")
        self.notify("cancellation requested; the workspace is preserved")

    async def action_quit(self) -> None:
        running = len(self.rt._running) if self.rt is not None else 0
        if running and time.monotonic() - self._quit_armed > 4:
            self._quit_armed = time.monotonic()
            self.notify(
                f"{running} task(s) running. Press ctrl+q again to cancel them and quit.", severity="warning"
            )
            return
        self.exit(return_code=0)


def run_tui(ctx: CLIContext, *, session_id: str | None = None) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        ctx.output.err.print(
            '[red]The TUI needs an interactive terminal.[/] Use `core run "<request>"` for non-interactive use.'
        )
        return 2
    app = CoreApp(ctx, session_id=session_id)
    app.run()
    return app.return_code or 0
