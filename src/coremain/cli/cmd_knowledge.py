"""`core index`, `core profile`, `core memory`, `core decisions`, `core skills`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from coremain.cli.common import CLIContext, runtime_command
from coremain.cli.output import ago
from coremain.errors import UsageError
from coremain.util.text import one_line


# ============================================================================ index
@click.group()
def index() -> None:
    """Codebase index: symbols, imports, search, impact."""


@index.command("build")
@click.option("--full", is_flag=True, help="Rebuild from scratch.")
@runtime_command()
async def index_build(ctx: CLIContext, rt: Any, full: bool) -> int:
    report = await rt.ensure_index(full=full)
    ctx.output.data(
        report,
        lambda c: c.print(
            f"scanned {report['scanned']} files: {report['changed']} (re)indexed, {report['unchanged']} unchanged, "
            f"{report['removed']} removed, {report['skipped']} skipped in {report['duration_ms']} ms"
        ),
    )
    return 0


@index.command("status")
@runtime_command()
async def index_status(ctx: CLIContext, rt: Any) -> int:
    st = rt.index.status(rt.require_project().id)
    ctx.output.data(
        st,
        lambda c: c.print(
            f"{st['files']} files, {st['symbols']} symbols, indexer v{st['indexer_version']} "
            f"(current v{st['current_version']}), last scan {ago(st['last_scan_at'])}\nlanguages: "
            + ", ".join(f"{k} {v}" for k, v in list(st["languages"].items())[:10])
        ),
    )
    return 0


@index.command("search")
@click.argument("query", nargs=-1, required=True)
@click.option("--limit", default=15, show_default=True)
@runtime_command()
async def index_search(ctx: CLIContext, rt: Any, query: tuple[str, ...], limit: int) -> int:
    hits = rt.index.search(rt.require_project().id, " ".join(query), limit=limit)
    rows = [h.__dict__ for h in hits]
    ctx.output.data(
        rows,
        lambda c: [
            c.print(f"[bold]{h.path}[/]:{h.start_line}-{h.end_line} [dim]{one_line(h.snippet, 120)}[/]")
            for h in hits
        ],
    )
    return 0


@index.command("symbols")
@click.argument("name")
@click.option("--exact", is_flag=True)
@runtime_command()
async def index_symbols(ctx: CLIContext, rt: Any, name: str, exact: bool) -> int:
    rows = rt.index.find_symbols(rt.require_project().id, name, exact=exact)
    ctx.output.data(
        rows,
        lambda c: ctx.output.table(
            ["location", "kind", "symbol", "signature"],
            [
                [f"{r['path']}:{r['line']}", r["kind"], r["qualname"], one_line(r["signature"] or "", 70)]
                for r in rows
            ],
        ),
    )
    return 0


@index.command("impact")
@click.argument("paths", nargs=-1, required=True)
@runtime_command()
async def index_impact(ctx: CLIContext, rt: Any, paths: tuple[str, ...]) -> int:
    data = rt.index.impact(rt.require_project().id, list(paths))
    ctx.output.data(
        data,
        lambda c: (
            c.print(
                "[bold]Affected:[/] "
                + (", ".join(f"{a['path']} (d{a['distance']})" for a in data["affected"]) or "none")
            ),
            c.print("[bold]Tests:[/] " + (", ".join(data["tests"]) or "none")),
        ),
    )
    return 0


@index.command("map")
@runtime_command()
async def index_map(ctx: CLIContext, rt: Any) -> int:
    text = rt.index.repo_map(rt.require_project().id, max_files=80)
    ctx.output.data({"map": text}, lambda c: c.print(text, markup=False))
    return 0


@click.command()
@click.option("--refresh", is_flag=True, help="Re-run intake even if manifests are unchanged.")
@runtime_command()
async def profile(ctx: CLIContext, rt: Any, refresh: bool) -> int:
    """Show the project intake profile (ecosystems, frameworks, commands, entry points)."""
    data = await rt.profile_for(rt.require_project(), refresh=refresh)
    ctx.output.data(data, lambda c: c.print(data.get("summary") or "(empty profile)", markup=False))
    return 0


# ============================================================================ memory
@click.group()
def memory() -> None:
    """Layered project memory with provenance (inspect, correct, invalidate, delete)."""


def _mem_rows(items: list[Any]) -> list[list[Any]]:
    return [
        [i.id, i.scope, i.kind, i.source_type, f"{i.confidence:.2f}", i.status, one_line(i.content, 70)]
        for i in items
    ]


@memory.command("list")
@click.option("--scope", type=click.Choice(["task", "session", "project", "operational", "global"]))
@click.option(
    "--kind",
    type=click.Choice(
        ["fact", "hypothesis", "suggestion", "preference", "decision", "observation", "procedure"]
    ),
)
@click.option("--all", "all_status", is_flag=True, help="Include invalidated/superseded entries.")
@runtime_command()
async def memory_list(ctx: CLIContext, rt: Any, scope: str | None, kind: str | None, all_status: bool) -> int:
    statuses = ("active", "stale", "invalidated", "superseded") if all_status else ("active", "stale")
    items = rt.memory.list(rt.require_project().id, scope=scope, kind=kind, statuses=statuses)
    ctx.output.data(
        [i.to_dict() for i in items],
        lambda c: ctx.output.table(
            ["id", "scope", "kind", "source", "conf", "status", "content"], _mem_rows(items)
        ),
    )
    return 0


@memory.command("search")
@click.argument("query", nargs=-1, required=True)
@runtime_command()
async def memory_search(ctx: CLIContext, rt: Any, query: tuple[str, ...]) -> int:
    items = rt.memory.search(rt.require_project().id, " ".join(query), limit=20)
    ctx.output.data(
        [i.to_dict() for i in items], lambda c: [c.print(f"{i.id} {i.label()} {i.content}") for i in items]
    )
    return 0


@memory.command("add")
@click.argument("content", nargs=-1, required=True)
@click.option(
    "--kind",
    type=click.Choice(["fact", "preference", "decision", "procedure", "observation"]),
    default="fact",
    show_default=True,
)
@click.option(
    "--scope",
    type=click.Choice(["project", "operational", "session", "global"]),
    default="project",
    show_default=True,
)
@click.option("--tag", "tags", multiple=True)
@runtime_command()
async def memory_add(
    ctx: CLIContext, rt: Any, content: tuple[str, ...], kind: str, scope: str, tags: tuple[str, ...]
) -> int:
    project = rt.require_project()
    session = rt.sessions.latest(project.id) if scope == "session" else None
    item = rt.memory.add(
        content=" ".join(content),
        kind=kind,
        scope=scope,
        source_type="user",
        project_id=project.id,
        session_id=session.id if session else None,
        tags=tags,
    )
    ctx.output.data(item.to_dict(), f"stored {item.id} {item.label()}")
    return 0


@memory.command("show")
@click.argument("memory_id")
@runtime_command()
async def memory_show(ctx: CLIContext, rt: Any, memory_id: str) -> int:
    item = rt.memory.resolve(memory_id)
    ctx.output.data(item.to_dict())
    return 0


@memory.command("invalidate")
@click.argument("memory_id")
@click.option("--reason", required=True)
@runtime_command()
async def memory_invalidate(ctx: CLIContext, rt: Any, memory_id: str, reason: str) -> int:
    item = rt.memory.invalidate(rt.memory.resolve(memory_id).id, reason=reason)
    ctx.output.data(item.to_dict(), f"invalidated {item.id}")
    return 0


@memory.command("promote")
@click.argument("memory_id")
@runtime_command()
async def memory_promote(ctx: CLIContext, rt: Any, memory_id: str) -> int:
    """Confirm a hypothesis/observation as a fact (user confirmation)."""
    item = rt.memory.promote(rt.memory.resolve(memory_id).id, by_user=True)
    ctx.output.data(item.to_dict(), f"promoted {item.id} to fact")
    return 0


@memory.command("delete")
@click.argument("memory_id")
@runtime_command()
async def memory_delete(ctx: CLIContext, rt: Any, memory_id: str) -> int:
    item = rt.memory.resolve(memory_id)
    rt.memory.delete(item.id)
    ctx.output.data({"deleted": item.id}, f"deleted {item.id}")
    return 0


@memory.command("check")
@runtime_command()
async def memory_check(ctx: CLIContext, rt: Any) -> int:
    """Mark memories whose anchored files changed as stale."""
    project = rt.require_project()
    stale = rt.memory.check_anchors(project.id, Path(project.root_path))
    ctx.output.data({"stale": stale}, f"{len(stale)} memory item(s) marked stale")
    return 0


# ============================================================================ decisions
@click.group()
def decisions() -> None:
    """Architecture decision records."""


@decisions.command("list")
@click.option("--status", type=click.Choice(["proposed", "accepted", "superseded", "rejected"]))
@runtime_command()
async def decisions_list(ctx: CLIContext, rt: Any, status: str | None) -> int:
    items = rt.decisions.list(rt.require_project().id, status=status)
    ctx.output.data(
        [d.to_dict() for d in items],
        lambda c: ctx.output.table(
            ["id", "status", "title", "source", "created"],
            [[d.id, d.status, one_line(d.title, 60), d.source, ago(d.created_at)] for d in items],
        ),
    )
    return 0


@decisions.command("show")
@click.argument("decision_id")
@runtime_command()
async def decisions_show(ctx: CLIContext, rt: Any, decision_id: str) -> int:
    d = rt.decisions.resolve(decision_id)
    ctx.output.data(d.to_dict(), lambda c: c.print(d.render(), markup=False))
    return 0


@decisions.command("add")
@click.option("--title", required=True)
@click.option("--context", "context_text", required=True)
@click.option("--decision", "decision_text", required=True)
@click.option("--alt", "alternatives", multiple=True, help="Rejected alternative (repeatable).")
@click.option("--consequences", default=None)
@click.option("--supersedes", default=None)
@runtime_command()
async def decisions_add(
    ctx: CLIContext,
    rt: Any,
    title: str,
    context_text: str,
    decision_text: str,
    alternatives: tuple[str, ...],
    consequences: str | None,
    supersedes: str | None,
) -> int:
    d = rt.decisions.add(
        rt.require_project().id,
        title=title,
        context=context_text,
        decision=decision_text,
        alternatives=alternatives,
        consequences=consequences,
        supersedes_id=rt.decisions.resolve(supersedes).id if supersedes else None,
    )
    ctx.output.data(d.to_dict(), f"recorded {d.id}")
    return 0


@decisions.command("accept")
@click.argument("decision_id")
@runtime_command()
async def decisions_accept(ctx: CLIContext, rt: Any, decision_id: str) -> int:
    d = rt.decisions.set_status(rt.decisions.resolve(decision_id).id, "accepted")
    ctx.output.data(d.to_dict(), f"accepted {d.id}")
    return 0


@decisions.command("reject")
@click.argument("decision_id")
@runtime_command()
async def decisions_reject(ctx: CLIContext, rt: Any, decision_id: str) -> int:
    d = rt.decisions.set_status(rt.decisions.resolve(decision_id).id, "rejected")
    ctx.output.data(d.to_dict(), f"rejected {d.id}")
    return 0


# ============================================================================ skills
@click.group()
def skills() -> None:
    """Skills: discover, validate, trust, import, scaffold."""


@skills.command("list")
@runtime_command(require_project=False)
async def skills_list(ctx: CLIContext, rt: Any) -> int:
    items = rt.skills.all()
    ctx.output.data(
        [s.to_dict() for s in items],
        lambda c: (
            ctx.output.table(
                ["name", "version", "source", "trust", "valid", "tokens", "description"],
                [
                    [
                        s.name,
                        s.version,
                        s.source,
                        s.trust,
                        "yes" if s.valid else "NO",
                        s.tokens,
                        one_line(s.description, 60),
                    ]
                    for s in items
                ],
            ),
            [c.print(f"[yellow]collision:[/] {x}") for x in rt.skills.collisions],
        ),
    )
    return 0


@skills.command("show")
@click.argument("name")
@runtime_command(require_project=False)
async def skills_show(ctx: CLIContext, rt: Any, name: str) -> int:
    s = rt.skills.get(name)
    ctx.output.data(
        s.to_dict(include_body=True),
        lambda c: (
            c.print(f"[bold]{s.name}[/] v{s.version} · {s.source} · {s.trust} · sha256 {s.sha256[:16]}"),
            c.print(s.description),
            ctx.output.markdown(s.body),
        ),
    )
    return 0


@skills.command("validate")
@click.argument("name", required=False)
@runtime_command(require_project=False)
async def skills_validate(ctx: CLIContext, rt: Any, name: str | None) -> int:
    items = [rt.skills.get(name)] if name else rt.skills.all()
    rows = [
        {"name": s.name, "valid": s.valid, "problems": s.problems, "warnings": s.warnings, "trust": s.trust}
        for s in items
    ]
    bad = [r for r in rows if not r["valid"]]
    ctx.output.data(
        rows,
        lambda c: [
            c.print(
                f"{'[green]ok[/]' if r['valid'] else '[red]invalid[/]'} {r['name']} ({r['trust']})"
                + "".join(f"\n  [red]problem:[/] {p}" for p in r["problems"])
                + "".join(f"\n  [yellow]warning:[/] {w}" for w in r["warnings"])
            )
            for r in rows
        ],
    )
    return 1 if bad else 0


@skills.command("trust")
@click.argument("name")
@runtime_command(require_project=False)
async def skills_trust(ctx: CLIContext, rt: Any, name: str) -> int:
    s = rt.skills.set_trust(name, "trusted")
    ctx.output.data(s.to_dict(), f"trusted {s.name} at sha256 {s.sha256[:16]} (any change revokes trust)")
    return 0


@skills.command("untrust")
@click.argument("name")
@runtime_command(require_project=False)
async def skills_untrust(ctx: CLIContext, rt: Any, name: str) -> int:
    s = rt.skills.set_trust(name, "untrusted")
    ctx.output.data(s.to_dict(), f"{s.name} is now {s.trust}")
    return 0


@skills.command("block")
@click.argument("name")
@runtime_command(require_project=False)
async def skills_block(ctx: CLIContext, rt: Any, name: str) -> int:
    s = rt.skills.set_trust(name, "blocked")
    ctx.output.data(s.to_dict(), f"blocked {s.name}")
    return 0


@skills.command("import")
@click.argument("source")
@click.option("--path", "subpath", default=None, help="Sub-directory inside the source containing skills.")
@click.option("--name", "names", multiple=True, help="Only import these skill names.")
@runtime_command(require_project=False)
async def skills_import(
    ctx: CLIContext, rt: Any, source: str, subpath: str | None, names: tuple[str, ...]
) -> int:
    """Import skills from a git URL or local directory (imported skills start untrusted)."""
    import asyncio

    if source.startswith(("http://", "https://")) and rt.config.permissions.network.offline:
        raise UsageError("offline mode: cannot fetch remote skills")
    imported = await asyncio.to_thread(
        rt.skills.import_skills,
        source,
        rt.paths.imported_skills_dir,
        subpath=subpath,
        names=list(names) or None,
    )
    ctx.output.data(
        [s.to_dict() for s in imported],
        lambda c: (
            [
                c.print(
                    f"imported {s.name} ({s.trust}; license {s.frontmatter.get('license') or 'unstated'}) → {s.path}"
                )
                for s in imported
            ]
            or c.print("no valid skills found")
        ),
    )
    return 0 if imported else 1


@skills.command("new")
@click.argument("name")
@click.option("--description", required=True)
@click.option(
    "--project",
    "in_project",
    is_flag=True,
    help="Create in .core/skills (project) instead of the user skills dir.",
)
@runtime_command(require_project=False)
async def skills_new(ctx: CLIContext, rt: Any, name: str, description: str, in_project: bool) -> int:
    from coremain.paths import ProjectPaths
    from coremain.skills.registry import NAME_RE

    if not NAME_RE.match(name):
        raise UsageError("skill names use lowercase letters, digits and single hyphens")
    base = (
        ProjectPaths(Path(rt.require_project().root_path)).skills_dir
        if in_project
        else rt.paths.user_skills_dir
    )
    target = base / name
    if target.exists():
        raise UsageError(f"{target} already exists")
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        f'---\nname: {name}\ndescription: {description}\nmetadata:\n  core-version: "1"\n  core-triggers: ""\n  core-kinds: ""\n'
        f'  core-stage: "any"\n---\n\n# {name}\n\nDescribe when to use this skill and the procedure to follow.\n',
        encoding="utf-8",
    )
    rt.skills.refresh()
    ctx.output.data(
        {"path": str(target)},
        f"created {target / 'SKILL.md'}"
        + (" (project skills require `core skills trust`)" if in_project else ""),
    )
    return 0


@skills.command("select")
@click.argument("text", nargs=-1, required=True)
@click.option("--kind", default=None, help="Task kind (default: classify the request like a real task).")
@click.option(
    "--stage", default=None, help="Workflow stage (default: the first stage of the chosen workflow)."
)
@runtime_command(require_project=False, detect_project=True)
async def skills_select(
    ctx: CLIContext, rt: Any, text: tuple[str, ...], kind: str | None, stage: str | None
) -> int:
    """Explain which skills would be selected for a request."""
    from coremain.routing.modes import classify

    request = " ".join(text)
    cls = classify(request)
    first_stage = {
        "debug": "debugging",
        "review": "review",
        "answer": "question",
        "plan": "planning",
        "collaborative": "architecture",
    }.get(cls.workflow, "implementation")
    kind = kind or cls.kind
    stage = stage or first_stage
    profile = await rt.profile_for(rt.project) if rt.project is not None else {}
    picks = rt.skills.select(
        text=request,
        task_kind=kind,
        languages=list(profile.get("languages", {}))[:4],
        frameworks=[f.split(" ")[0] for f in profile.get("frameworks", [])],
        stage=stage,
    )
    rows = [
        {"name": p.skill.name, "score": round(p.score, 2), "reasons": p.reasons, "kind": kind, "stage": stage}
        for p in picks
    ]
    ctx.output.data(
        rows,
        lambda c: (
            c.print(f"[dim]kind {kind} · stage {stage}[/]"),
            [c.print(f"{r['name']} ({r['score']}): {'; '.join(r['reasons'])}") for r in rows]
            or c.print("no skills selected"),
        ),
    )
    return 0
