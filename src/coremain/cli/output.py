"""Consistent human and machine-readable output for every command."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from coremain.domain.states import TaskStatus
from coremain.errors import CoreError, ExitCode
from coremain.util.jsonutil import dumps


def task_exit_code(status: TaskStatus | str) -> ExitCode:
    s = TaskStatus(status)
    if s == TaskStatus.COMPLETED:
        return ExitCode.OK
    if s == TaskStatus.BLOCKED:
        return ExitCode.BLOCKED
    if s == TaskStatus.FAILED:
        return ExitCode.FAILURE
    if s == TaskStatus.CANCELLED:
        return ExitCode.CANCELLED
    return ExitCode.INCOMPLETE


@dataclass
class Output:
    json_mode: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        self.console = Console(stderr=False, highlight=False, soft_wrap=False)
        self.err = Console(stderr=True, highlight=False)

    def data(self, payload: Any, human: Callable[[Console], object] | str | None = None) -> None:
        if self.json_mode:
            sys.stdout.write(dumps({"ok": True, "data": payload}, indent=2) + "\n")
            return
        if human is None:
            self.console.print_json(dumps(payload))
        elif isinstance(human, str):
            self.console.print(human)
        else:
            human(self.console)

    def jsonl(self, obj: Any) -> None:
        sys.stdout.write(dumps(obj) + "\n")
        sys.stdout.flush()

    def error(self, exc: BaseException) -> ExitCode:
        if isinstance(exc, CoreError):
            code = exc.exit_code
            payload = exc.to_dict()
        else:
            code = ExitCode.INTERNAL
            payload = {"class": "internal_error", "message": f"{type(exc).__name__}: {exc}"}
        if self.json_mode:
            sys.stdout.write(dumps({"ok": False, "error": payload, "exit_code": int(code)}, indent=2) + "\n")
        else:
            self.err.print(f"[bold red]error[/] [dim]({payload['class']})[/]: {payload['message']}")
            if payload.get("hint"):
                self.err.print(f"[yellow]hint:[/] {payload['hint']}")
            details = payload.get("details") or {}
            if self.verbose and details:
                self.err.print_json(dumps(details))
        return code

    def markdown(self, text: str) -> None:
        if self.json_mode:
            return
        self.console.print(Markdown(text))

    def table(self, columns: list[str], rows: Iterable[Iterable[Any]], title: str | None = None) -> None:
        table = Table(title=title, show_lines=False, header_style="bold", expand=False)
        for col in columns:
            table.add_column(col, overflow="fold")
        for row in rows:
            table.add_row(*[str(c) if c is not None else "" for c in row])
        self.console.print(table)


def ago(ts: float | None) -> str:
    if not ts:
        return ""
    delta = max(0, time.time() - ts)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{int(delta // size)}{unit} ago"
    return f"{int(delta)}s ago"


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str)
