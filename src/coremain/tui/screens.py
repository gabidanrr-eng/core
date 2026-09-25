"""Modal screens: approvals, agent questions and help."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown, Static

from coremain.tools.base import ApprovalDecision
from coremain.ui.describe import esc

HELP = """\
# Core Main

Type a request and press **Enter**. Core Main classifies it, picks an execution mode, works in
an isolated workspace, verifies the result with real evidence and only then applies it.

## Commands

| Command | Effect |
|---|---|
| `/ask <question>` | Read-only question about the repository (answer mode) |
| `/mode <mode\\|auto>` | Force a mode for new requests: direct, plan, collaborative, adversarial, deep, debug, answer, review |
| `/model <key\\|auto>` | Pin a configured model for new requests |
| `/cancel` | Cancel the current task (durable; workspace preserved) |
| `/resume` | Resume the current interrupted/paused task |
| `/diff` | Show the current task's changes |
| `/apply` | Apply a completed task's workspace to your working tree |
| `/approve`, `/deny` | Decide the oldest pending approval |
| `/new [title]` | Start a new session |
| `/tasks` | List recent tasks in this session |
| `/help` | This help |
| `/quit` | Quit (running tasks are cancelled after confirmation) |

## Keys

`ctrl+p` command palette · `ctrl+n` new session · `ctrl+b` toggle sessions · `ctrl+k` cancel task ·
`f2`–`f6` task / evidence / diff / activity / approvals · `f1` help · `ctrl+q` quit
"""


class ApprovalScreen(ModalScreen[ApprovalDecision]):
    BINDINGS = [
        Binding("y", "decide('once')", "Allow once"),
        Binding("s", "decide('session')", "Allow for session"),
        Binding("n,escape", "deny", "Deny"),
    ]

    def __init__(self, approval: Any):
        super().__init__()
        self.approval = approval

    def compose(self) -> ComposeResult:
        req = self.approval.request or {}
        with Vertical(id="dialog", classes="approval"):
            yield Label("Approval needed", id="dialog-title")
            yield Static(
                f"[b]{esc(self.approval.capability)}[/b]  {esc(self.approval.summary)}", id="dialog-summary"
            )
            details = [
                f"[dim]{key}:[/] {esc(req[key])}"
                for key in ("reason", "risk", "tool", "target", "cwd")
                if req.get(key)
            ]
            if details:
                yield Static("\n".join(details), id="dialog-details")
            with Horizontal(id="dialog-buttons"):
                yield Button("Allow once [y]", id="once", variant="primary")
                yield Button("This session [s]", id="session")
                yield Button("Deny [n]", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "deny":
            self.action_deny()
        else:
            self.action_decide(event.button.id or "once")

    def action_decide(self, scope: str) -> None:
        self.dismiss(ApprovalDecision(True, scope))

    def action_deny(self) -> None:
        self.dismiss(ApprovalDecision(False, "once", "denied in the TUI"))


class QuestionScreen(ModalScreen[str]):
    BINDINGS = [Binding("escape", "skip", "Skip")]

    def __init__(self, question: str, context: str | None):
        super().__init__()
        self.question = question
        self.context = context

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="question"):
            yield Label("The agent asks", id="dialog-title")
            yield Static(esc(self.question), id="dialog-summary")
            if self.context:
                yield Static(f"[dim]{esc(self.context)}[/]", id="dialog-details")
            yield Input(placeholder="Your answer (Enter to send, Esc to skip)", id="answer")

    def on_mount(self) -> None:
        self.query_one("#answer", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_skip(self) -> None:
        self.dismiss("")


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,f1,q", "close", "Close")]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog", classes="help"):
            yield Markdown(HELP)

    def action_close(self) -> None:
        self.dismiss(None)
