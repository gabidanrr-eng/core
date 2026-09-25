"""Command palette entries (ctrl+p), mirroring the slash commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from textual.command import DiscoveryHit, Hit, Hits, Provider

if TYPE_CHECKING:
    from coremain.tui.app import CoreApp

STATIC = [
    ("New session", "/new", "Start a new session"),
    ("Cancel current task", "/cancel", "Durably cancel the running task"),
    ("Resume current task", "/resume", "Resume an interrupted or paused task"),
    ("Show diff", "/diff", "Changes made by the current task"),
    ("Apply current task", "/apply", "Apply a completed task's workspace to the working tree"),
    ("Approve pending approval", "/approve", "Allow the oldest pending approval once"),
    ("Deny pending approval", "/deny", "Deny the oldest pending approval"),
    ("List tasks", "/tasks", "Recent tasks in this session"),
    ("Mode: automatic", "/mode auto", "Let Core Main choose the execution mode"),
    ("Model: automatic", "/model auto", "Let the router choose models"),
    ("Help", "/help", "Commands and keys"),
]
MODES = ["direct", "plan", "collaborative", "adversarial", "deep", "debug", "answer", "review"]


class CoreCommands(Provider):
    def _entries(self) -> list[tuple[str, str, str]]:
        app: CoreApp = self.app  # type: ignore[assignment]
        entries = list(STATIC)
        entries += [(f"Mode: {m}", f"/mode {m}", f"Force the {m} mode for new requests") for m in MODES]
        if app.rt is not None:
            entries += [
                (f"Model: {m.key}", f"/model {m.key}", f"Pin {m.ref} for new requests")
                for m in app.rt.registry.models()
            ]
        return entries

    def _runner(self, command: str) -> Callable[[], Any]:
        app: CoreApp = self.app  # type: ignore[assignment]
        return lambda: app.call_later(app.run_slash, command)

    async def discover(self) -> Hits:
        for name, command, help_text in self._entries():
            yield DiscoveryHit(name, self._runner(command), help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, command, help_text in self._entries():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), self._runner(command), help=help_text)
