"""Operational commands: mcp, browser, eval, learn, db, export/import, metrics, logs, doctor, selftest, recover, gc, serve, research."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import click

from coremain.cli.common import CLIContext, CommandExit, run_async, runtime_command
from coremain.cli.output import ago
from coremain.config.edit import set_in, unset_in
from coremain.errors import ConfigError, ExitCode, NotFoundError, UsageError
from coremain.paths import resolve_core_paths
from coremain.util.jsonutil import dumps
from coremain.util.text import one_line


# ============================================================================ mcp
@click.group()
def mcp() -> None:
    """MCP servers: configure, check, list tools."""


@mcp.command("list")
@runtime_command(require_project=False)
async def mcp_list(ctx: CLIContext, rt: Any) -> int:
    manager = rt.extension("mcp")
    rows = manager.describe()
    ctx.output.data(
        rows,
        lambda c: (
            ctx.output.table(
                ["server", "transport", "trust", "enabled", "status", "tools", "checked", "error"],
                [
                    [
                        r["name"],
                        r["transport"],
                        r["trust"],
                        r["enabled"],
                        r["status"],
                        r["tools"],
                        ago(r["checked_at"]),
                        one_line(r.get("error") or "", 60),
                    ]
                    for r in rows
                ],
            )
            if rows
            else c.print("No MCP servers configured. Add one with `core mcp add`.")
        ),
    )
    return 0


@mcp.command("add")
@click.argument("name")
@click.option("--url", default=None, help="Streamable HTTP endpoint.")
@click.option(
    "--env", "env_pairs", multiple=True, help="KEY=VALUE environment for stdio servers (non-secret)."
)
@click.option(
    "--secret-env",
    "secret_env_pairs",
    multiple=True,
    help="KEY=REF (env:/store:/file:/command:) for secrets.",
)
@click.option("--header", "headers", multiple=True, help="Header: Value (non-secret).")
@click.option(
    "--secret-header",
    "secret_headers",
    multiple=True,
    help="Header=REF for secret headers (e.g. Authorization=store:github).",
)
@click.option(
    "--trust", "trusted", is_flag=True, help="Mark the server trusted (read-only tools auto-allowed)."
)
@click.option("--description", default=None)
@click.argument("command", nargs=-1)
@click.pass_obj
def mcp_add(
    ctx: CLIContext,
    name: str,
    url: str | None,
    env_pairs: tuple[str, ...],
    secret_env_pairs: tuple[str, ...],
    headers: tuple[str, ...],
    secret_headers: tuple[str, ...],
    trusted: bool,
    description: str | None,
    command: tuple[str, ...],
) -> None:
    """Add a server: `core mcp add NAME -- npx -y @playwright/mcp@latest --headless` or `core mcp add NAME --url https://...`."""
    from coremain.cli.cmd_config import _validated_update

    if bool(url) == bool(command):
        raise CommandExit(int(ctx.output.error(UsageError("give either a command (after --) or --url"))))
    entry: dict[str, Any] = {
        "transport": "http" if url else "stdio",
        "trust": "trusted" if trusted else "untrusted",
    }
    if url:
        entry["url"] = url
    else:
        entry["command"] = list(command)
    if description:
        entry["description"] = description

    def pairs(items: tuple[str, ...], sep: str) -> dict[str, str]:
        out = {}
        for item in items:
            k, _, v = item.partition(sep)
            if not k or not _:
                raise CommandExit(int(ctx.output.error(UsageError(f"expected KEY{sep}VALUE, got '{item}'"))))
            out[k.strip()] = v.strip()
        return out

    for key, items, sep in (
        ("env", env_pairs, "="),
        ("secret_env", secret_env_pairs, "="),
        ("headers", headers, ":"),
        ("secret_headers", secret_headers, "="),
    ):
        if items:
            entry[key] = pairs(items, sep)
    path = resolve_core_paths(ctx.env).config_file
    try:
        _validated_update(ctx, path, lambda d: set_in(d, f"mcp.{name}", entry))
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data(
        {"server": name, "config": entry},
        f"added MCP server [bold]{name}[/]; verify with `core mcp check {name}`",
    )


@mcp.command("remove")
@click.argument("name")
@click.pass_obj
def mcp_remove(ctx: CLIContext, name: str) -> None:
    from coremain.cli.cmd_config import _validated_update

    path = resolve_core_paths(ctx.env).config_file
    try:
        _validated_update(ctx, path, lambda d: unset_in(d, f"mcp.{name}"))
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data({"removed": name}, f"removed MCP server {name}")


@mcp.command("check")
@click.argument("name", required=False)
@runtime_command(require_project=False)
async def mcp_check(ctx: CLIContext, rt: Any, name: str | None) -> int:
    manager = rt.extension("mcp")
    names = [name] if name else list(rt.config.mcp)
    results = []
    for n in names:
        results.append(await manager.check(n))
    worst = 0 if all(r["status"] == "ready" for r in results) else int(ExitCode.BLOCKED)
    ctx.output.data(
        results,
        lambda c: (
            [
                c.print(
                    f"{r['name']}: {r['status']}"
                    + (
                        f" (protocol {r.get('protocol_version')}, {r.get('tools', 0)} tools)"
                        if r["status"] == "ready"
                        else f" — {r.get('error_class')}: {r.get('error')}"
                    )
                )
                for r in results
            ]
            or c.print("no MCP servers configured")
        ),
    )
    return worst


@mcp.command("tools")
@click.argument("name")
@runtime_command(require_project=False)
async def mcp_tools(ctx: CLIContext, rt: Any, name: str) -> int:
    manager = rt.extension("mcp")
    tools = await manager.list_tools(name)
    ctx.output.data(
        tools,
        lambda c: [
            c.print(
                f"[bold]{t['name']}[/] {one_line(t.get('description') or '', 90)} [dim]{t.get('annotations') or ''}[/]"
            )
            for t in tools
        ],
    )
    return 0


@mcp.command("call")
@click.argument("name")
@click.argument("tool")
@click.option("--args", "args_json", default="{}", help="JSON arguments.")
@runtime_command(require_project=False)
async def mcp_call(ctx: CLIContext, rt: Any, name: str, tool: str, args_json: str) -> int:
    """Call an MCP tool directly (user-initiated; audited)."""
    manager = rt.extension("mcp")
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as exc:
        raise UsageError(f"--args is not valid JSON: {exc}") from exc
    result = await manager.call_tool(name, tool, args)
    ctx.output.data(result, lambda c: c.print(result.get("text") or dumps(result), markup=False))
    return 1 if result.get("is_error") else 0


# ============================================================================ browser
@click.group()
def browser() -> None:
    """Browser automation (Playwright)."""


@browser.command("check")
@click.option("--live", is_flag=True, help="Launch a real headless browser.")
@runtime_command(require_project=False)
async def browser_check(ctx: CLIContext, rt: Any, live: bool) -> int:
    manager = rt.extension("browser")
    data = await manager.check(live=live)
    ctx.output.data(
        data,
        lambda c: c.print(
            f"browser: {data['status']}" + (f" — {data.get('detail')}" if data.get("detail") else "")
        ),
    )
    return 0 if data["status"] in {"available", "ready"} else int(ExitCode.BLOCKED)


@browser.command("screenshot")
@click.argument("url")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("screenshot.png"),
    show_default=True,
)
@click.option("--width", default=1280, show_default=True)
@click.option("--height", default=800, show_default=True)
@runtime_command(require_project=False)
async def browser_screenshot(
    ctx: CLIContext, rt: Any, url: str, output: Path, width: int, height: int
) -> int:
    manager = rt.extension("browser")
    info = await manager.screenshot_url(url, output, width=width, height=height)
    ctx.output.data(info, f"saved {output} ({info.get('title', '')})")
    return 0


# ============================================================================ research
@click.group()
def research() -> None:
    """Documentation retrieval (Context7-style) and web fetch, policy-controlled."""


@research.command("docs")
@click.argument("library")
@click.argument("query", nargs=-1)
@runtime_command(require_project=False)
async def research_docs(ctx: CLIContext, rt: Any, library: str, query: tuple[str, ...]) -> int:
    service = rt.extension("research")
    result = await service.docs(library, " ".join(query) or library)
    ctx.output.data(
        result,
        lambda c: (
            c.print(f"[bold]{result['library_id']}[/] ({result['source']})"),
            c.print(result["text"][:6000], markup=False),
        ),
    )
    return 0


@research.command("fetch")
@click.argument("url")
@runtime_command(require_project=False)
async def research_fetch(ctx: CLIContext, rt: Any, url: str) -> int:
    service = rt.extension("research")
    result = await service.fetch(url)
    ctx.output.data(result, lambda c: c.print(result["text"][:6000], markup=False))
    return 0


# ============================================================================ eval
@click.group("eval")
def eval_group() -> None:
    """Evaluation harness: replay benchmark scenarios against strategies and measure outcomes."""


@eval_group.command("list")
@click.pass_obj
def eval_list(ctx: CLIContext) -> None:
    from coremain.evals.scenarios import load_scenarios

    scenarios = load_scenarios()
    ctx.output.data(
        [s.to_dict() for s in scenarios],
        lambda c: ctx.output.table(
            ["id", "family", "suite", "title"],
            [[s.id, s.family, ",".join(s.suites), s.title] for s in scenarios],
        ),
    )


@eval_group.command("run")
@click.option("--suite", default="smoke", show_default=True)
@click.option("--scenario", "scenario_ids", multiple=True)
@click.option("--model", default=None, help="Pin a model key for all roles.")
@click.option("--mode", default=None)
@click.option("--label", default=None)
@click.option("--keep", is_flag=True, help="Keep scenario workspaces for inspection.")
@runtime_command(require_project=False)
async def eval_run(
    ctx: CLIContext,
    rt: Any,
    suite: str,
    scenario_ids: tuple[str, ...],
    model: str | None,
    mode: str | None,
    label: str | None,
    keep: bool,
) -> int:
    from coremain.evals.harness import EvalHarness

    harness = EvalHarness(rt)
    run = await harness.run(
        suite=suite,
        scenario_ids=list(scenario_ids),
        model=model,
        mode=mode,
        label=label,
        keep=keep,
        on_result=lambda r: (
            ctx.output.console.print(
                f"  {'[green]pass' if r['status'] == 'pass' else '[red]' + r['status']}[/] "
                f"{r['scenario']} [dim]({r['task_status']}, {r['metrics'].get('duration_s')}s)[/]"
            )
            if not ctx.output.json_mode
            else None
        ),
    )
    ctx.output.data(
        run,
        lambda c: c.print(
            f"run {run['id']}: {run['summary']['passed']}/{run['summary']['total']} passed "
            f"({run['summary']['pass_rate']:.0%}) · strategy {run['strategy']}"
        ),
    )
    return 0 if run["summary"]["failed"] == 0 and run["summary"]["errors"] == 0 else 1


@eval_group.command("report")
@click.option("--run", "run_id", default=None)
@click.option("--suite", default=None)
@runtime_command(require_project=False)
async def eval_report(ctx: CLIContext, rt: Any, run_id: str | None, suite: str | None) -> int:
    from coremain.evals.report import build_report

    report = build_report(rt, run_id=run_id, suite=suite)
    ctx.output.data(report, lambda c: c.print(report["text"], markup=False))
    return 0


# ============================================================================ learn
@click.group()
def learn() -> None:
    """Learning loop: failures → patterns → versioned heuristics → validation → activation."""


@learn.command("analyze")
@runtime_command(require_project=False)
async def learn_analyze(ctx: CLIContext, rt: Any) -> int:
    from coremain.learning.analysis import analyze

    result = analyze(rt, propose=rt.config.learning.auto_propose_heuristics)
    ctx.output.data(result, lambda c: c.print(result["text"], markup=False))
    return 0


@learn.command("list")
@runtime_command(require_project=False)
async def learn_list(ctx: CLIContext, rt: Any) -> int:
    rows = [
        dict(r)
        for r in rt.db.query(
            "SELECT id, name, version, kind, status, rationale, updated_at FROM heuristics ORDER BY updated_at DESC"
        )
    ]
    ctx.output.data(
        rows,
        lambda c: ctx.output.table(
            ["id", "name", "v", "kind", "status", "rationale"],
            [
                [r["id"], r["name"], r["version"], r["kind"], r["status"], one_line(r["rationale"], 60)]
                for r in rows
            ],
        ),
    )
    return 0


@learn.command("validate")
@click.argument("heuristic_id")
@click.option("--suite", default="smoke", show_default=True)
@runtime_command(require_project=False)
async def learn_validate(ctx: CLIContext, rt: Any, heuristic_id: str, suite: str) -> int:
    from coremain.learning.analysis import validate_heuristic

    result = await validate_heuristic(rt, heuristic_id, suite=suite)
    ctx.output.data(result, lambda c: c.print(f"{heuristic_id}: {result['status']} — {result['reason']}"))
    return 0 if result["status"] == "validated" else 1


@learn.command("activate")
@click.argument("heuristic_id")
@click.option("--force", is_flag=True, help="Activate without a passing validation (recorded).")
@runtime_command(require_project=False)
async def learn_activate(ctx: CLIContext, rt: Any, heuristic_id: str, force: bool) -> int:
    from coremain.learning.analysis import set_heuristic_status

    row = set_heuristic_status(rt, heuristic_id, "active", force=force)
    ctx.output.data(row, f"activated {row['name']} v{row['version']}")
    return 0


@learn.command("reject")
@click.argument("heuristic_id")
@runtime_command(require_project=False)
async def learn_reject(ctx: CLIContext, rt: Any, heuristic_id: str) -> int:
    from coremain.learning.analysis import set_heuristic_status

    row = set_heuristic_status(rt, heuristic_id, "rejected")
    ctx.output.data(row, f"rejected {row['name']} v{row['version']}")
    return 0


@learn.command("rollback")
@click.argument("name")
@runtime_command(require_project=False)
async def learn_rollback(ctx: CLIContext, rt: Any, name: str) -> int:
    from coremain.learning.analysis import rollback

    row = rollback(rt, name)
    ctx.output.data(row, f"rolled back {name}; now active: {row.get('active_version') or 'none'}")
    return 0


@learn.command("regressions")
@runtime_command(require_project=False)
async def learn_regressions(ctx: CLIContext, rt: Any) -> int:
    rows = [
        dict(r)
        for r in rt.db.query(
            "SELECT id, failure_id, title, status, created_at FROM regression_cases ORDER BY created_at DESC"
        )
    ]
    candidates = [
        dict(r)
        for r in rt.db.query(
            "SELECT id, category, error_class, summary FROM failures WHERE regression_candidate = 1 AND id NOT IN "
            "(SELECT failure_id FROM regression_cases WHERE failure_id IS NOT NULL) ORDER BY created_at DESC LIMIT 20"
        )
    ]
    ctx.output.data(
        {"cases": rows, "candidates": candidates},
        lambda c: (
            ctx.output.table(
                ["id", "failure", "title", "status"],
                [[r["id"], r["failure_id"], r["title"], r["status"]] for r in rows],
            ),
            [
                c.print(
                    f"candidate {r['id']}: {r['category']}/{r['error_class']} {one_line(r['summary'], 80)} "
                    f"(`core learn regress {r['id']}`)"
                )
                for r in candidates
            ],
        ),
    )
    return 0


@learn.command("regress")
@click.argument("failure_id")
@runtime_command(require_project=False)
async def learn_regress(ctx: CLIContext, rt: Any, failure_id: str) -> int:
    """Turn a resolved failure into a durable regression case."""
    from coremain.learning.analysis import create_regression_case

    case = create_regression_case(rt, failure_id)
    ctx.output.data(case, f"created regression case {case['id']}")
    return 0


# ============================================================================ db / export / import
@click.group()
def db() -> None:
    """Database maintenance: integrity, backup/restore, migrations."""


@db.command("check")
@click.option("--full", is_flag=True)
@runtime_command(require_project=False)
async def db_check(ctx: CLIContext, rt: Any, full: bool) -> int:
    from coremain.store.migrate import status as mig_status

    problems = rt.db.integrity_check(full=full)
    st = mig_status(rt.db)
    data = {
        "integrity": problems or "ok",
        "schema": st.current,
        "latest": st.latest,
        "pending": st.pending,
        "checksum_mismatches": st.checksum_mismatches,
        "artifact_problems": rt.artifacts.verify(limit=500),
    }
    bad = bool(problems or st.checksum_mismatches or data["artifact_problems"])
    ctx.output.data(
        data,
        lambda c: c.print(
            f"integrity {data['integrity']}; schema v{st.current}/{st.latest}; "
            f"artifact problems: {len(data['artifact_problems'])}"
        ),
    )
    return 1 if bad else 0


@db.command("backup")
@click.argument("path", required=False, type=click.Path(dir_okay=False, path_type=Path))
@runtime_command(require_project=False)
async def db_backup(ctx: CLIContext, rt: Any, path: Path | None) -> int:
    target = path or rt.paths.backups_dir / f"core-{time.strftime('%Y%m%d-%H%M%S')}.db"
    rt.db.backup_to(target)
    ctx.output.data({"backup": str(target)}, f"backup written to {target}")
    return 0


@db.command("restore")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--yes", is_flag=True)
@click.pass_obj
def db_restore(ctx: CLIContext, path: Path, yes: bool) -> None:
    """Replace the database with a backup (refuses while other runtimes are active)."""
    from coremain.store.backup import restore_database

    if not yes and not click.confirm(
        f"Replace the current database with {path}? (the current one is backed up first)", default=False
    ):
        raise CommandExit(int(ExitCode.CANCELLED))
    try:
        info = restore_database(resolve_core_paths(ctx.env), path)
    except Exception as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data(info, f"restored {path}; previous database saved to {info['previous_backup']}")


@db.command("migrate")
@runtime_command(require_project=False)
async def db_migrate(ctx: CLIContext, rt: Any) -> int:
    from coremain.store.migrate import status as mig_status

    st = mig_status(rt.db)
    ctx.output.data(
        {"schema": st.current, "latest": st.latest},
        f"schema v{st.current} (latest v{st.latest}); migrations apply automatically on open",
    )
    return 0


@db.command("vacuum")
@runtime_command(require_project=False)
async def db_vacuum(ctx: CLIContext, rt: Any) -> int:
    before = rt.paths.db_path.stat().st_size
    rt.db.checkpoint()
    rt.db.execute("VACUUM")
    after = rt.paths.db_path.stat().st_size
    ctx.output.data({"before": before, "after": after}, f"database {before} → {after} bytes")
    return 0


@click.command("export")
@click.option("--session", "session_id", default=None)
@click.option("--task", "task_id", default=None)
@click.option(
    "--project",
    "whole_project",
    is_flag=True,
    help="Export all sessions, tasks, memory and decisions of this project.",
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--artifacts/--no-artifacts", default=False, help="Embed referenced text artifacts (redacted).")
@runtime_command()
async def export_cmd(
    ctx: CLIContext,
    rt: Any,
    session_id: str | None,
    task_id: str | None,
    whole_project: bool,
    output: Path,
    artifacts: bool,
) -> int:
    """Export sessions/tasks/evidence/memory as a versioned, redacted JSON bundle."""
    from coremain.store.backup import export_bundle

    if not (session_id or task_id or whole_project):
        raise UsageError("choose --session, --task or --project")
    info = export_bundle(
        rt,
        output,
        session_id=session_id,
        task_id=task_id,
        whole_project=whole_project,
        include_artifacts=artifacts,
    )
    ctx.output.data(info, f"exported {info['counts']} to {output}")
    return 0


@click.command("import")
@click.argument("bundle", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@runtime_command()
async def import_cmd(ctx: CLIContext, rt: Any, bundle: Path) -> int:
    """Import a bundle created by `core export` into the current project."""
    from coremain.store.backup import import_bundle

    info = import_bundle(rt, bundle)
    ctx.output.data(info, f"imported {info['counts']} (skipped existing: {info['skipped']})")
    return 0


# ============================================================================ observability
@click.command()
@click.option("--days", default=30, show_default=True)
@runtime_command(require_project=False)
async def metrics(ctx: CLIContext, rt: Any, days: int) -> int:
    """Latency, failure and usage metrics by provider/model/role and tool."""
    from coremain.observe.metrics import collect

    data = collect(rt, days=days)
    ctx.output.data(data, lambda c: c.print(data["text"], markup=False))
    return 0


@click.command()
@click.option("--task", "task_id", default=None)
@click.option("--kind", default=None, help="Event kind prefix filter, e.g. 'tool.'")
@click.option("--limit", default=50, show_default=True)
@click.option("--follow", is_flag=True, help="Keep printing new events (cross-process tail).")
@runtime_command(require_project=False)
async def logs(
    ctx: CLIContext, rt: Any, task_id: str | None, kind: str | None, limit: int, follow: bool
) -> int:
    """Tail durable execution events."""
    sql = (
        "SELECT seq FROM events"
        + (" WHERE task_id = ?" if task_id else "")
        + " ORDER BY seq DESC LIMIT 1 OFFSET ?"
    )
    start_row = rt.db.one(sql, ((task_id, limit) if task_id else (limit,)))
    seq = int(start_row["seq"]) if start_row else 0
    try:
        while True:
            events = rt.events.since(seq, task_id=task_id, limit=500)
            for ev in events:
                seq = ev.seq or seq
                if kind and not ev.kind.startswith(kind):
                    continue
                if ctx.output.json_mode:
                    ctx.output.jsonl(ev.to_dict())
                else:
                    ctx.output.console.print(
                        f"[dim]{time.strftime('%H:%M:%S', time.localtime(ev.ts))} {ev.seq}[/] {ev.kind:<24} "
                        f"[dim]{(ev.task_id or '')[-8:]}[/] {one_line(dumps(ev.data), 120)}",
                        markup=True,
                    )
            if not follow:
                break
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    return 0


# ============================================================================ diagnostics / maintenance
@click.command()
@click.option("--live", is_flag=True, help="Also contact providers and MCP servers (no tokens consumed).")
@runtime_command(require_project=False)
async def doctor(ctx: CLIContext, rt: Any, live: bool) -> int:
    """Check database, migrations, config, providers, MCP, skills, index, workspaces, processes and caches."""
    from coremain.diagnostics.doctor import run_doctor

    report = await run_doctor(rt, live=live)
    icons = {"ok": "[green]✓[/]", "warn": "[yellow]![/]", "fail": "[red]✗[/]", "skip": "[dim]-[/]"}
    ctx.output.data(
        report,
        lambda c: (
            [
                c.print(
                    f"{icons[ch['status']]} {ch['name']:<28} {ch['detail']}"
                    + (f"\n    [yellow]→ {ch['fix']}[/]" if ch.get("fix") else "")
                )
                for ch in report["checks"]
            ],
            c.print(f"\n{report['summary']}"),
        ),
    )
    return 1 if report["failed"] else 0


@click.command()
@click.option(
    "--live",
    is_flag=True,
    help="Also run one real model call if a provider is configured and credentials are present.",
)
@click.option("--keep", is_flag=True, help="Keep the temporary self-test directory.")
@click.pass_obj
def selftest(ctx: CLIContext, live: bool, keep: bool) -> None:
    """Verify Core Main itself in an isolated temporary environment."""
    from coremain.diagnostics.selftest import run_selftest

    report = run_async(
        ctx,
        run_selftest(
            live=live,
            keep=keep,
            user_env=ctx.env,
            progress=None
            if ctx.output.json_mode
            else lambda name, ok, detail: ctx.output.console.print(
                f"{'[green]✓' if ok else '[red]✗'}[/] {name} [dim]{detail}[/]"
            ),
        ),
    )
    ctx.output.data(
        report,
        lambda c: c.print(
            f"\nself-test: {report['passed']}/{report['total']} checks passed"
            + (f"; kept {report['dir']}" if keep else "")
        ),
    )
    if report["failed"]:
        raise CommandExit(1)


@click.command("recover")
@click.option("--kill-orphans", is_flag=True, help="Terminate orphaned processes left by dead runtimes.")
@click.option("--dry-run", is_flag=True)
@runtime_command(require_project=False)
async def recover_cmd(ctx: CLIContext, rt: Any, kill_orphans: bool, dry_run: bool) -> int:
    """Reconcile durable state after crashes: stale leases, interrupted attempts, orphans."""
    from coremain.runtime.recovery import recover

    report = recover(rt, kill_orphans=kill_orphans, dry_run=dry_run).to_dict()
    if rt.last_recovery is not None and not dry_run:
        first = rt.last_recovery.to_dict()
        for key in ("stale_runtimes", "interrupted_tasks", "unknown_tasks"):
            report[key] = [*first.get(key, []), *report.get(key, [])]
    ctx.output.data(
        report,
        lambda c: [c.print(f"{k}: {v}") for k, v in report.items() if v] or c.print("nothing to recover"),
    )
    return 0


@click.command()
@click.option("--apply", "do_apply", is_flag=True, help="Actually delete (default is a dry run).")
@click.option("--force-orphans", is_flag=True, help="Also delete workspace directories without records.")
@click.option(
    "--artifact-days",
    default=30,
    show_default=True,
    help="Expire unprotected artifacts of finished tasks older than N days.",
)
@runtime_command(require_project=False)
async def gc(ctx: CLIContext, rt: Any, do_apply: bool, force_orphans: bool, artifact_days: int) -> int:
    """Garbage-collect workspaces and artifacts without touching active or recoverable work."""
    from coremain.domain.states import TERMINAL

    statuses = {r["id"]: r["status"] for r in rt.db.query("SELECT id, status FROM tasks")}
    keep_tasks = {tid for tid, st in statuses.items() if st not in {s.value for s in TERMINAL}}
    ws_actions = await rt.workspaces.gc(
        task_status=statuses, dry_run=not do_apply, force_orphans=force_orphans
    )
    art = rt.artifacts.gc(max_age_s=artifact_days * 86400, keep_task_ids=keep_tasks, dry_run=not do_apply)
    data = {
        "dry_run": not do_apply,
        "workspaces": [a.__dict__ for a in ws_actions],
        "artifacts": art.__dict__,
    }
    ctx.output.data(
        data,
        lambda c: (
            [c.print(f"{a.action:<7} {a.path} [dim]({a.reason})[/]") for a in ws_actions],
            c.print(
                f"artifacts: expire {art.expired_rows}, remove {art.removed_blobs} blobs ({art.freed_bytes} bytes), "
                f"{art.removed_temp} temp files; protected {art.protected}"
            ),
            c.print("[dim](dry run — pass --apply to delete)[/]" if not do_apply else ""),
        ),
    )
    return 0


@click.command()
@click.option(
    "--stdio", is_flag=True, required=True, help="Serve the runtime API as JSON-RPC 2.0 over stdin/stdout."
)
@click.pass_obj
def serve(ctx: CLIContext, stdio: bool) -> None:
    """Expose the runtime to editors/other clients (JSON-RPC over stdio)."""
    from coremain.server.jsonrpc import serve_stdio

    run_async(ctx, serve_stdio(ctx))


__all__ = ["NotFoundError"]
