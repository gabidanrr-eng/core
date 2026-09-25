"""Tools over codebase intelligence, memory and skills."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field

from coremain.errors import ToolError
from coremain.security.paths import resolve_within
from coremain.security.policy import Capability
from coremain.tools.base import Tool, ToolContext, ToolInput, ToolResult
from coremain.tools.fs import guard


def _index(ctx: ToolContext):  # type: ignore[no-untyped-def]
    if ctx.services.intel is None:
        raise ToolError("code index is unavailable", error_class="unavailable")
    return ctx.services.intel


class FindSymbol(Tool):
    name = "find_symbol"
    description = "Find definitions of a symbol (function, class, method, type, route) in the code index."
    capability = Capability.FS_READ

    class Input(ToolInput):
        name: str = Field(min_length=1)
        exact: bool = False

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        rows = _index(ctx).find_symbols(ctx.project_id, args.name, exact=args.exact, limit=40)
        if not rows:
            return ToolResult(True, f"no symbols matching '{args.name}' (index may be stale; try search_text)")
        lines = [f"{r['path']}:{r['line']} {r['kind']} {r['qualname']}" + (f" — {r['signature']}" if r["signature"] else "") for r in rows]
        return ToolResult(True, "\n".join(lines), data={"count": len(rows)})


class FileOutline(Tool):
    name = "file_outline"
    description = "Show the structure of a file (classes, functions, signatures with line numbers) without reading it fully."
    capability = Capability.FS_READ

    class Input(ToolInput):
        path: str

    def target(self, args: Input) -> str:
        return args.path

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        _, rel = guard(ctx, args.path)
        text = _index(ctx).outline_text(ctx.project_id, rel)
        return ToolResult(True, text or f"no indexed symbols for {rel}")


class FindReferences(Tool):
    name = "find_references"
    description = "Find textual references to an identifier across the workspace (word-boundary match)."
    capability = Capability.FS_READ

    class Input(ToolInput):
        identifier: str = Field(min_length=2, pattern=r"^[A-Za-z_$][\w$.]*$")
        max_results: int = Field(80, ge=1, le=500)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        from coremain.tools.fs import SearchText

        return await SearchText().run(ctx, SearchText.Input(pattern=rf"\b{args.identifier.replace('.', chr(92) + '.')}\b", regex=True,
                                                            case_sensitive=True, max_results=args.max_results))


class ImpactAnalysis(Tool):
    name = "impact_analysis"
    description = "Given changed files, list files that import them (transitively) and related tests."
    capability = Capability.FS_READ

    class Input(ToolInput):
        paths: list[str] = Field(min_length=1)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        rels = [guard(ctx, p)[1] for p in args.paths]
        return ToolResult(True, json.dumps(_index(ctx).impact(ctx.project_id, rels), indent=1))


class RepoMap(Tool):
    name = "repo_map"
    description = "Summarize repository structure: key files ranked by importance with their main symbols."
    capability = Capability.FS_READ

    class Input(ToolInput):
        focus: list[str] = Field(default_factory=list)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        return ToolResult(True, _index(ctx).repo_map(ctx.project_id, focus=args.focus, max_files=60))


class ProjectProfileTool(Tool):
    name = "project_profile"
    description = "Show the detected project profile: ecosystems, frameworks, test/lint/build commands, entry points."
    capability = Capability.FS_READ

    class Input(ToolInput):
        pass

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        return ToolResult(True, json.dumps(ctx.services.profile, indent=1)[:12000])


class MemorySearch(Tool):
    name = "memory_search"
    description = "Search project memory (facts, decisions, observations, hypotheses) with provenance labels."
    capability = Capability.FS_READ

    class Input(ToolInput):
        query: str

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        mem = ctx.services.memory
        if mem is None:
            raise ToolError("memory is disabled", error_class="unavailable")
        items = mem.search(ctx.project_id, args.query, session_id=ctx.session_id, limit=15)
        if not items:
            return ToolResult(True, "no relevant memory")
        return ToolResult(True, "\n".join(f"{i.id} {i.label()} {i.content}" for i in items))


class MemoryPropose(Tool):
    name = "memory_propose"
    description = ("Propose a durable project memory. Model proposals are stored as hypotheses/observations/suggestions; "
                   "they become facts only when confirmed by the user or linked evidence.")
    capability = Capability.MEMORY_WRITE

    class Input(ToolInput):
        content: str = Field(min_length=10, max_length=2000)
        kind: Literal["hypothesis", "observation", "suggestion"] = "observation"
        scope: Literal["task", "session", "project", "operational"] = "project"
        tags: list[str] = Field(default_factory=list)
        anchor_paths: list[str] = Field(default_factory=list, description="Files this memory depends on (for staleness detection)")

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        from coremain.util.jsonutil import sha256_hex

        mem = ctx.services.memory
        if mem is None:
            raise ToolError("memory is disabled", error_class="unavailable")
        anchors = []
        for p in args.anchor_paths[:10]:
            path, rel = guard(ctx, p)
            if path.is_file():
                anchors.append({"path": rel, "sha256": sha256_hex(path.read_bytes())})
        item = mem.add(content=args.content, kind=args.kind, scope=args.scope, source_type="model", project_id=ctx.project_id,
                       session_id=ctx.session_id, task_id=ctx.task_id, tags=args.tags, source_ref=f"task:{ctx.task_id}", anchors=anchors)
        note = f" ({'; '.join(item.notes)})" if item.notes else ""
        return ToolResult(True, f"stored {item.id} as {item.kind}{note}")


class LoadSkill(Tool):
    name = "load_skill"
    description = "Load the full instructions of an available skill by name (see the skills catalog in context)."
    capability = Capability.FS_READ

    class Input(ToolInput):
        name: str

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        reg = ctx.services.skills
        if reg is None:
            raise ToolError("skills are disabled", error_class="unavailable")
        skill = reg.get(args.name)
        if not skill.usable:
            raise ToolError(f"skill '{args.name}' is {skill.trust}{'' if skill.valid else ' and invalid'}; it cannot be loaded",
                            error_class="untrusted_skill")
        ctx.loaded_skills.add(skill.name)
        ctx.services.events.emit("skill.loaded", project_id=ctx.project_id, task_id=ctx.task_id, data={"skill": skill.name, "version": skill.version})
        res = ("\nResources (read with read_skill_resource): " + ", ".join(skill.resources[:30])) if skill.resources else ""
        return ToolResult(True, f"# Skill: {skill.name} (v{skill.version})\n{skill.body}{res}")


class ReadSkillResource(Tool):
    name = "read_skill_resource"
    description = "Read a text resource bundled with a loaded skill (references/, templates, scripts)."
    capability = Capability.FS_READ

    class Input(ToolInput):
        skill: str
        path: str

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        reg = ctx.services.skills
        if reg is None:
            raise ToolError("skills are disabled", error_class="unavailable")
        skill = reg.get(args.skill)
        if not skill.usable:
            raise ToolError(f"skill '{args.skill}' is not usable ({skill.trust})", error_class="untrusted_skill")
        target = resolve_within(skill.path, args.path)
        if not target.is_file():
            raise ToolError(f"{args.path} not found in skill {args.skill}", error_class="not_found")
        data = target.read_bytes()
        if b"\x00" in data[:4096]:
            return ToolResult(True, f"[binary resource, {len(data)} bytes]")
        return ToolResult(True, data.decode("utf-8", errors="replace")[:40_000])


KNOWLEDGE_TOOLS: list[type[Tool]] = [FindSymbol, FileOutline, FindReferences, ImpactAnalysis, RepoMap, ProjectProfileTool, MemorySearch,
                                     MemoryPropose, LoadSkill, ReadSkillResource]
