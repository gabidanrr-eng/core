"""Conversation, per-task live blocks, session list and status bar."""

from __future__ import annotations

import time
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Label, ListItem, ListView, Markdown, Static
from textual.widgets.markdown import MarkdownStream

from coremain.ui.describe import esc, styled_status
from coremain.util.text import one_line

MAX_ACTIVITY_LINES = 14


def ago(ts: float | None) -> str:
    if not ts:
        return ""
    delta = max(0, time.time() - ts)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{int(delta // size)}{unit} ago"
    return "just now"


class TaskBlock(Vertical):
    """One task inside the conversation: status header, recent activity, streamed text, report."""

    def __init__(self, task_id: str, title: str, status: str = "queued"):
        super().__init__(classes="task-block")
        self.task_id = task_id
        self.title = title
        self.status = status
        self.reason = ""
        self._lines: list[str] = []
        self._stream: MarkdownStream | None = None
        self.done = False
        self._finishing = False

    def compose(self) -> ComposeResult:
        yield Static(self._header(), classes="task-header")
        yield Static("", classes="task-activity")
        yield Markdown("", classes="task-live")
        yield Markdown("", classes="task-report")

    def _header(self) -> str:
        reason = f" [dim]{esc(one_line(self.reason, 90))}[/]" if self.reason else ""
        return f"[b]{esc(one_line(self.title, 80))}[/b]  {styled_status(self.status)}{reason}  [dim]{self.task_id[-8:]}[/]"

    def set_status(self, status: str, reason: str | None = None) -> None:
        self.status = status
        self.reason = reason or ""
        self.query_one(".task-header", Static).update(self._header())

    def add_line(self, markup: str) -> None:
        self._lines.append(markup)
        del self._lines[:-MAX_ACTIVITY_LINES]
        self.query_one(".task-activity", Static).update("\n".join(self._lines))

    async def stream(self, text: str) -> None:
        if self.done or self._finishing:
            return
        live = self.query_one(".task-live", Markdown)
        if self._stream is None:
            live.display = True
            self._stream = Markdown.get_stream(live)
        await self._stream.write(text)

    async def reset_live(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
        await self.query_one(".task-live", Markdown).update("")

    async def show_report(self, markdown: str) -> None:
        self._finishing = True
        await self.reset_live()
        self.query_one(".task-live", Markdown).display = False
        await self.query_one(".task-report", Markdown).update(markdown)
        self.done = True

    def reopen(self) -> None:
        """The task continues (approval/input/resume): accept live output again."""
        self.done = False
        self._finishing = False


class Conversation(VerticalScroll):
    def __init__(self, **kw: Any):
        super().__init__(**kw)
        self.blocks: dict[str, TaskBlock] = {}

    async def clear(self) -> None:
        self.blocks.clear()
        await self.remove_children()

    async def add_user(self, text: str) -> None:
        await self.mount(Static(esc(text), classes="user-msg"))
        self.scroll_end(animate=False)

    async def add_markdown(self, text: str, classes: str = "assistant-msg") -> None:
        await self.mount(Markdown(text, classes=classes))
        self.scroll_end(animate=False)

    async def add_note(self, markup: str, classes: str = "note") -> None:
        await self.mount(Static(markup, classes=classes))
        self.scroll_end(animate=False)

    async def add_task(self, task_id: str, title: str, status: str) -> TaskBlock:
        block = TaskBlock(task_id, title, status)
        self.blocks[task_id] = block
        await self.mount(block)
        self.scroll_end(animate=False)
        return block


class SessionItem(ListItem):
    def __init__(self, session: Any):
        super().__init__(Label(f"{esc(one_line(session.title, 26))}\n[dim]{ago(session.updated_at)}[/]"))
        self.session_id = session.id


class SessionList(ListView):
    pass


class StatusBar(Static):
    def render_status(
        self,
        *,
        project: str,
        session: str,
        mode: str | None,
        model: str | None,
        running: int,
        offline: bool,
        models: int,
    ) -> None:
        parts = [
            f"[b]Core Main[/b] · {esc(project)}",
            f"session [b]{esc(one_line(session, 30))}[/b]",
            f"mode {esc(mode or 'auto')}",
            f"model {esc(model or 'auto')}",
        ]
        if running:
            parts.append(f"[cyan]{running} running[/]")
        if offline:
            parts.append("[yellow]offline[/]")
        if not models:
            parts.append("[red]no models configured[/]")
        self.update("  ·  ".join(parts))
