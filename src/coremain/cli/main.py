"""`core` command-line entry point."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import click

from coremain.cli.common import CLIContext, CommandExit, run_async, runtime_command
from coremain.cli.output import Output, ago, task_exit_code
from coremain.errors import CoreError, ExitCode, UsageError
from coremain.ui.describe import styled_status
from coremain.version import __version__

MODES = ["direct", "plan", "collaborative", "adversarial", "deep", "debug", "answer", "review", "recovery"]


class LazyGroup(click.Group):
    """Top-level group whose sub-groups are imported on demand (fast startup)."""

    lazy: dict[str, tuple[str, str]] = {
        "session": ("coremain.cli.cmd_tasks", "session"),
        "task": ("coremain.cli.cmd_tasks", "task"),
        "approvals": ("coremain.cli.cmd_tasks", "approvals"),
        "config": ("coremain.cli.cmd_config", "config"),
        "permissions": ("coremain.cli.cmd_config", "permissions"),
        "providers": ("coremain.cli.cmd_config", "providers"),
        "models": ("coremain.cli.cmd_config", "models"),
        "index": ("coremain.cli.cmd_knowledge", "index"),
        "profile": ("coremain.cli.cmd_knowledge", "profile"),
        "memory": ("coremain.cli.cmd_knowledge", "memory"),
        "decisions": ("coremain.cli.cmd_knowledge", "decisions"),
        "skills": ("coremain.cli.cmd_knowledge", "skills"),
        "mcp": ("coremain.cli.cmd_ops", "mcp"),
        "browser": ("coremain.cli.cmd_ops", "browser"),
        "eval": ("coremain.cli.cmd_ops", "eval_group"),
        "learn": ("coremain.cli.cmd_ops", "learn"),
        "db": ("coremain.cli.cmd_ops", "db"),
        "export": ("coremain.cli.cmd_ops", "export_cmd"),
        "import": ("coremain.cli.cmd_ops", "import_cmd"),
        "metrics": ("coremain.cli.cmd_ops", "metrics"),
        "logs": ("coremain.cli.cmd_ops", "logs"),
        "doctor": ("coremain.cli.cmd_ops", "doctor"),
        "selftest": ("coremain.cli.cmd_ops", "selftest"),
        "recover": ("coremain.cli.cmd_ops", "recover_cmd"),
        "gc": ("coremain.cli.cmd_ops", "gc"),
        "serve": ("coremain.cli.cmd_ops", "serve"),
        "research": ("coremain.cli.cmd_ops", "research"),
    }

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self.lazy})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.lazy:
            import importlib

            module, attr = self.lazy[cmd_name]
            return getattr(importlib.import_module(module), attr)  # type: ignore[no-any-return]
        return super().get_command(ctx, cmd_name)


@click.group(
    cls=LazyGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 110},
)
@click.option("--json", "json_mode", is_flag=True, help="Machine-readable JSON output.")
@click.option(
    "--project",
    "project",
    type=click.Path(file_okay=False, path_type=Path),
    help="Project directory (default: nearest repo root).",
)
@click.option(
    "--home",
    type=click.Path(file_okay=False, path_type=Path),
    help="Relocate all Core Main state (sets CORE_HOME).",
)
@click.option(
    "--permission-profile",
    type=click.Choice(["read-only", "standard", "autonomous"]),
    help="Permission profile for this invocation.",
)
@click.option("--offline", is_flag=True, help="Disable all network access for this invocation.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output (stream model text, show details).")
@click.version_option(__version__, prog_name="core-main")
@click.pass_context
def cli(
    ctx: click.Context,
    json_mode: bool,
    project: Path | None,
    home: Path | None,
    permission_profile: str | None,
    offline: bool,
    verbose: bool,
) -> None:
    """Core Main — a terminal AI engineering environment with durable, evidence-based execution.

    Run `core` inside a repository to open the TUI, or `core run "<request>"` for non-interactive use.
    """
    env = dict(os.environ)
    if home:
        env["CORE_HOME"] = str(home.expanduser().resolve())
    overrides: dict[str, Any] = {}
    if permission_profile:
        overrides.setdefault("permissions", {})["profile"] = permission_profile
    if offline:
        overrides.setdefault("permissions", {}).setdefault("network", {})["offline"] = True
    ctx.obj = CLIContext(
        Output(json_mode=json_mode, verbose=verbose), project=project, overrides=overrides, env=env
    )
    if ctx.invoked_subcommand is None:
        if sys.stdout.isatty() and sys.stdin.isatty() and not json_mode:
            ctx.invoke(tui)
        else:
            ctx.invoke(status)


# --------------------------------------------------------------------------- run / ask
def _load_contract(
    path: str | None,
    accept: tuple[str, ...],
    constraint: tuple[str, ...],
    forbid: tuple[str, ...],
    require_cmd: tuple[str, ...],
    require_test: tuple[str, ...],
) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    if path:
        from coremain.config.loader import read_toml

        contract.update(read_toml(Path(path)))
    for key, values in (
        ("acceptance_criteria", accept),
        ("constraints", constraint),
        ("forbidden_paths", forbid),
        ("required_commands", require_cmd),
        ("required_tests", require_test),
    ):
        if values:
            contract[key] = [*contract.get(key, []), *values]
    return contract


async def _run_request(
    ctx: CLIContext,
    rt: Any,
    text: str,
    *,
    mode: str | None,
    model: str | None,
    direct: bool,
    no_apply: bool,
    session_id: str | None,
    new_session: bool,
    contract: dict[str, Any],
    non_interactive: bool,
    jsonl: bool,
    detach: bool,
) -> int:
    from coremain.cli.progress import ProgressPrinter, event_filter, terminal_approval, terminal_input

    out = ctx.output
    await rt.start()
    interactive = not non_interactive and sys.stdin.isatty() and not out.json_mode and not jsonl
    if interactive:
        rt.approval_handler = lambda approval: terminal_approval(out, approval)
        rt.input_handler = lambda q, c: terminal_input(out, q, c)
    if new_session:
        session_id = rt.ensure_session(new=True).id
    task = await rt.submit(
        text,
        session_id=session_id,
        mode=mode,
        model=model,
        workspace_mode="direct" if direct else None,
        contract=contract,
        auto_apply=False if no_apply else None,
    )
    if detach:
        out.data(
            {"task": task.to_dict()},
            f"created task [bold]{task.id}[/] ({task.mode}); run it with `core task resume {task.id} --run`",
        )
        return 0
    ids = {task.id}
    sub = rt.bus.subscribe(event_filter(ids))
    printer = ProgressPrinter(out, jsonl=jsonl, verbose=out.verbose)
    if not jsonl and not out.json_mode:
        out.console.print(f"[bold]Task[/] {task.id} [dim]· session {task.session_id}[/]")
    printer_task = asyncio.create_task(printer.consume(sub)) if not out.json_mode else None
    runner = rt.start_task(task.id)
    try:
        final = await asyncio.shield(runner)
    except asyncio.CancelledError:
        rt.cancel_task(task.id, reason="interrupted from the terminal (Ctrl-C)")
        try:
            final = await asyncio.wait_for(runner, timeout=30)
        except (TimeoutError, asyncio.CancelledError):
            final = rt.tasks.get(task.id)
    await asyncio.sleep(0.05)
    if printer_task is not None:
        printer_task.cancel()
    sub.close()
    report = next(
        (
            m.content
            for m in reversed(rt.sessions.messages(final.session_id, limit=20))
            if m.task_id == final.id and m.role == "assistant"
        ),
        final.result_summary or "",
    )
    if out.json_mode:
        evidence = [e.to_dict() for e in rt.evidence.for_task(final.id)]
        out.data(
            {
                "task": final.to_dict(),
                "report": report,
                "evidence": evidence,
                "findings": rt.reviews.open_findings(final.id),
                "exit_code": int(task_exit_code(final.status)),
            }
        )
    elif jsonl:
        out.jsonl({"kind": "final", "task": final.to_dict(), "report": report})
    else:
        out.console.print()
        out.markdown(report)
        if final.status.value not in {"completed"}:
            out.console.print(f"\nstatus: {styled_status(final.status.value)} — {final.status_reason or ''}")
    return int(task_exit_code(final.status))


def _run_options(fn: Any) -> Any:
    options = [
        click.option("--mode", type=click.Choice(MODES), help="Override the execution mode."),
        click.option("--model", help="Model key to use (pins planner/implementer)."),
        click.option(
            "--direct",
            is_flag=True,
            help="Modify the working tree directly (explicit authorization) instead of an isolated worktree.",
        ),
        click.option(
            "--no-apply", is_flag=True, help="Keep verified changes in the task workspace; do not apply them."
        ),
        click.option("--session", "session_id", help="Session id (default: latest active)."),
        click.option("--new-session", is_flag=True, help="Start a new session."),
        click.option("--accept", multiple=True, help="Acceptance criterion (repeatable)."),
        click.option("--constraint", multiple=True, help="Constraint (repeatable)."),
        click.option("--forbid", multiple=True, help="Glob of paths that must not change (repeatable)."),
        click.option(
            "--require-cmd", multiple=True, help="Command that must pass for completion (repeatable)."
        ),
        click.option("--require-test", multiple=True, help="Test path/selector that must pass (repeatable)."),
        click.option(
            "--contract",
            "contract_file",
            type=click.Path(exists=True, dir_okay=False),
            help="TOML task contract.",
        ),
        click.option(
            "--non-interactive", is_flag=True, help="Never prompt; approvals suspend the task (per policy)."
        ),
        click.option("--jsonl", is_flag=True, help="Stream execution events as JSON lines."),
        click.option("--detach", is_flag=True, help="Only create the task."),
    ]
    for opt in reversed(options):
        fn = opt(fn)
    return fn


@cli.command()
@click.argument("request", nargs=-1, required=True)
@_run_options
@runtime_command()
async def run(
    ctx: CLIContext,
    rt: Any,
    request: tuple[str, ...],
    mode: str | None,
    model: str | None,
    direct: bool,
    no_apply: bool,
    session_id: str | None,
    new_session: bool,
    accept: tuple[str, ...],
    constraint: tuple[str, ...],
    forbid: tuple[str, ...],
    require_cmd: tuple[str, ...],
    require_test: tuple[str, ...],
    contract_file: str | None,
    non_interactive: bool,
    jsonl: bool,
    detach: bool,
) -> int:
    """Execute an engineering request end to end (plan → implement → verify → review → evidence)."""
    contract = _load_contract(contract_file, accept, constraint, forbid, require_cmd, require_test)
    return await _run_request(
        ctx,
        rt,
        " ".join(request),
        mode=mode,
        model=model,
        direct=direct,
        no_apply=no_apply,
        session_id=session_id,
        new_session=new_session,
        contract=contract,
        non_interactive=non_interactive,
        jsonl=jsonl,
        detach=detach,
    )


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--model", help="Model key to use.")
@click.option("--session", "session_id")
@click.option("--jsonl", is_flag=True)
@runtime_command()
async def ask(
    ctx: CLIContext,
    rt: Any,
    question: tuple[str, ...],
    model: str | None,
    session_id: str | None,
    jsonl: bool,
) -> int:
    """Ask a read-only question about the repository (answers cite the files actually read)."""
    return await _run_request(
        ctx,
        rt,
        " ".join(question),
        mode="answer",
        model=model,
        direct=False,
        no_apply=True,
        session_id=session_id,
        new_session=False,
        contract={},
        non_interactive=True,
        jsonl=jsonl,
        detach=False,
    )


# --------------------------------------------------------------------------- status
@cli.command()
@runtime_command(require_project=False)
async def status(ctx: CLIContext, rt: Any) -> int:
    """Show project, session, task, model and integration state."""
    data = rt.status()

    def human(c: Any) -> None:
        c.print(f"[bold]Core Main[/] {data['version']} · data {data['data_dir']}")
        proj = data.get("project")
        if proj:
            c.print(
                f"[bold]Project[/] {proj['name']} ({proj['root']}) · vcs {proj['vcs']} · config {'trusted' if proj['trusted'] else 'untrusted'}"
            )
            if proj.get("profile"):
                for line in proj["profile"].splitlines()[:6]:
                    c.print(f"  [dim]{line}[/]")
            if data.get("session"):
                c.print(f"[bold]Session[/] {data['session']['id']} — {data['session']['title']}")
            idx = data.get("index", {})
            c.print(
                f"[bold]Index[/] {idx.get('files', 0)} files, {idx.get('symbols', 0)} symbols"
                + (
                    f", scanned {ago(idx.get('last_scan_at'))}"
                    if idx.get("last_scan_at")
                    else " (not built; run `core index build`)"
                )
            )
            counts = data.get("tasks", {}).get("counts", {})
            if counts:
                c.print(
                    "[bold]Tasks[/] "
                    + ", ".join(f"{styled_status(k)} {v}" for k, v in sorted(counts.items()))
                )
            for t in data.get("tasks", {}).get("attention", [])[:8]:
                c.print(
                    f"  [yellow]![/] {t['id']} {styled_status(t['status'])} {t['title'][:70]} [dim]{(t.get('reason') or '')[:80]}[/]"
                )
            for a in data.get("approvals", [])[:8]:
                c.print(f"  [yellow]approval[/] {a['id']} {a['summary'][:90]}")
        else:
            c.print("[dim]No project (run inside a repository or pass --project).[/]")
        if data["models"]:
            c.print("[bold]Models[/] " + ", ".join(f"{m['key']} ({m['ref']})" for m in data["models"]))
        else:
            c.print(
                "[yellow]No models configured.[/] Add one: `core providers add openai` then `core models add ...` (see `core models add --help`)."
            )
        for p in data["providers"]:
            c.print(f"  provider {p['id']} [{p['kind']}] credential {p['credential']}")
        c.print(
            f"[bold]Permissions[/] profile {data['permissions']['profile']}"
            + (" · offline" if data["permissions"]["offline"] else "")
        )
        for w in data.get("config_warnings", []):
            c.print(f"[yellow]config:[/] {w}")

    ctx.output.data(data, human)
    return 0


# --------------------------------------------------------------------------- init / trust
CONFIG_TEMPLATE = """# Core Main project configuration (committable). Secrets never belong here.
# Precedence: defaults < ~/.config/coremain/config.toml < .core/config.toml < .core/config.local.toml < env < CLI.
# Until you run `core trust`, this file cannot add providers/MCP servers or relax permissions.

[verification]
# Commands are auto-detected; override them here if detection is wrong.
# commands = {{ test = "{test}" }}
min_level = "weak"        # weak | moderate | strong
require_review = true

[permissions]
# profile = "standard"    # read-only | standard | autonomous (project config may only make it stricter)

[context]
instruction_files = ["AGENTS.md", "CORE.md", "CLAUDE.md", ".core/instructions.md"]
"""


@cli.command()
@click.option("--trust", "trust_now", is_flag=True, help="Also trust the project configuration.")
@runtime_command()
async def init(ctx: CLIContext, rt: Any, trust_now: bool) -> int:
    """Initialize Core Main for this project: intake, profile and a starter .core/config.toml."""
    from coremain.paths import ProjectPaths

    project = rt.require_project()
    pp = ProjectPaths(Path(project.root_path))
    pp.core_dir.mkdir(exist_ok=True)
    created = []
    profile = await rt.profile_for(project, refresh=True)
    if not pp.config_file.exists():
        pp.config_file.write_text(
            CONFIG_TEMPLATE.format(test=(profile.get("commands") or {}).get("test", "pytest -q")),
            encoding="utf-8",
        )
        created.append(str(pp.config_file))
    gi = pp.core_dir / ".gitignore"
    if not gi.exists():
        gi.write_text("config.local.toml\ncache/\ntmp/\n", encoding="utf-8")
        created.append(str(gi))
    if trust_now:
        from coremain.config.loader import project_config_hash

        rt.projects.set_trusted(project.id, project_config_hash(pp.root))
    ctx.output.data(
        {"created": created, "profile": profile, "trusted": trust_now},
        lambda c: (
            c.print(f"[bold]Initialized[/] {project.name}; created: {', '.join(created) or 'nothing new'}"),
            c.print(profile.get("summary", "")),
        ),
    )
    return 0


@cli.command()
@click.option("--revoke", is_flag=True, help="Revoke trust.")
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
@runtime_command()
async def trust(ctx: CLIContext, rt: Any, revoke: bool, yes: bool) -> int:
    """Trust (or untrust) this project's .core configuration (pinned to its content hash)."""
    from coremain.config.loader import project_config_hash, read_toml
    from coremain.paths import ProjectPaths

    project = rt.require_project()
    if revoke:
        rt.projects.revoke_trust(project.id)
        ctx.output.data({"trusted": False}, "project trust revoked")
        return 0
    pp = ProjectPaths(Path(project.root_path))
    privileged = sorted(
        {
            k
            for f in (pp.config_file, pp.local_config_file)
            for k in read_toml(f)
            if k in {"providers", "mcp", "permissions", "exec", "research", "browser", "workspace", "intel"}
        }
    )
    if not yes and not ctx.output.json_mode:
        ctx.output.console.print(
            f"Project config sections with elevated effect: {', '.join(privileged) or 'none'}"
        )
        if not click.confirm("Trust this project's configuration?", default=False):
            return int(ExitCode.CANCELLED)
    config_hash = project_config_hash(pp.root)
    rt.projects.set_trusted(project.id, config_hash)
    ctx.output.data(
        {"trusted": True, "config_hash": config_hash, "privileged_sections": privileged},
        "project configuration trusted",
    )
    return 0


# --------------------------------------------------------------------------- misc
@cli.command()
@click.option("--session", "session_id", help="Open a specific session.")
@click.pass_obj
def tui(ctx: CLIContext, session_id: str | None) -> None:
    """Open the interactive terminal UI."""
    from coremain.tui.app import run_tui

    code = run_tui(ctx, session_id=session_id)
    if code:
        raise CommandExit(code)


@cli.command()
@click.pass_obj
def version(ctx: CLIContext) -> None:
    """Print version information."""
    import platform
    import sqlite3

    from coremain.agent.prompts import PROMPT_VERSION
    from coremain.store.migrate import load_migrations

    data = {
        "core_main": __version__,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "schema": max(m.version for m in load_migrations()),
        "prompts": PROMPT_VERSION,
    }
    ctx.output.data(data, lambda c: c.print(" · ".join(f"{k} {v}" for k, v in data.items())))


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str) -> None:
    """Print a shell completion script (e.g. `core completion bash >> ~/.bashrc`)."""
    from click.shell_completion import get_completion_class

    cls = get_completion_class(shell)
    if cls is None:
        raise click.UsageError(f"unsupported shell {shell}")
    comp = cls(cli, {}, "core", "_CORE_COMPLETE")
    click.echo(comp.source())


def main(argv: list[str] | None = None) -> None:
    try:
        cli.main(args=argv, prog_name="core", standalone_mode=False)
    except CommandExit as exc:
        sys.exit(exc.code)
    except click.exceptions.Abort:
        sys.exit(130)
    except click.ClickException as exc:
        exc.show()
        sys.exit(ExitCode.USAGE if isinstance(exc, click.UsageError) else exc.exit_code)
    except CoreError as exc:
        sys.exit(int(Output().error(exc)))
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(0)


__all__ = ["UsageError", "cli", "main", "run_async"]
