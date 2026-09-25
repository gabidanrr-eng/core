"""Shared CLI plumbing: context object, runtime lifecycle, error → exit-code mapping."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import update_wrapper
from pathlib import Path
from typing import Any, TypeVar

import click

from coremain.cli.output import Output
from coremain.errors import CoreError, ExitCode
from coremain.paths import find_project_root, resolve_core_paths

T = TypeVar("T")


@dataclass
class CLIContext:
    output: Output
    project: Path | None = None
    no_project: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    exit_code: int = 0

    def project_root(self) -> Path | None:
        if self.no_project:
            return None
        return (self.project or find_project_root()).resolve()

    def detected_project(self) -> Path | None:
        """The explicit project, or the nearest ancestor with ``.core``/``.git``; never a bare cwd."""
        if self.no_project:
            return None
        if self.project:
            return self.project.resolve()
        root = find_project_root()
        return root if (root / ".git").exists() or (root / ".core").is_dir() else None

    def open_runtime(
        self,
        *,
        mode: str = "cli",
        require_project: bool = True,
        auto_recover: bool = True,
        detect_project: bool = False,
    ) -> Any:
        from coremain.runtime.app import CoreRuntime

        if require_project or self.project:
            root = self.project_root()
        else:
            root = self.detected_project() if detect_project else None
        return CoreRuntime.open(
            project_root=root,
            paths=resolve_core_paths(self.env),
            overrides=self.overrides or None,
            env=self.env,
            mode=mode,
            auto_recover=auto_recover,
        )


class CommandExit(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


pass_ctx = click.make_pass_decorator(CLIContext)


def run_async(ctx: CLIContext, coro: Awaitable[T]) -> T:
    try:
        return asyncio.run(coro)  # type: ignore[arg-type]
    except CommandExit:
        raise
    except KeyboardInterrupt:
        ctx.output.err.print("[yellow]interrupted[/]")
        raise CommandExit(130) from None
    except CoreError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc


def runtime_command(
    *, require_project: bool = True, mode: str = "cli", detect_project: bool = False
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Any]]:
    """Decorator: open a CoreRuntime, call ``async fn(rt, **kwargs)``, close it, map errors to exit codes."""

    def decorate(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
        @click.pass_obj
        def wrapper(ctx: CLIContext, **kwargs: Any) -> Any:
            async def body() -> Any:
                rt = ctx.open_runtime(
                    mode=mode, require_project=require_project, detect_project=detect_project
                )
                try:
                    return await fn(ctx, rt, **kwargs)
                finally:
                    await rt.close()

            result = run_async(ctx, body())
            if isinstance(result, int) and result != 0:
                raise CommandExit(result)
            return result

        return update_wrapper(wrapper, fn)

    return decorate


def fail(ctx: CLIContext, exc: CoreError) -> None:
    raise CommandExit(int(ctx.output.error(exc)))


__all__ = ["CLIContext", "CommandExit", "ExitCode", "pass_ctx", "run_async", "runtime_command"]
