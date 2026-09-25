"""Git tools. Reads are always allowed; commits are local-only and policy-governed. There is
deliberately no push/publish tool: remote operations require an explicit user action."""

from __future__ import annotations

from pydantic import Field

from coremain.errors import ToolError
from coremain.security.policy import Capability
from coremain.tools.base import SideEffect, Tool, ToolContext, ToolInput, ToolResult
from coremain.tools.fs import guard
from coremain.workspaces.git import Git

_IDENT = ("-c", "user.name=Core Main", "-c", "user.email=core-main@localhost", "-c", "commit.gpgsign=false")


def _git(ctx: ToolContext) -> Git:
    return Git(ctx.workspace.path)


async def _out(ctx: ToolContext, *args: str) -> str:
    res = await _git(ctx).run(*args, check=False)
    if res.code != 0:
        raise ToolError(f"git {args[0]} failed: {res.stderr[:400]}", error_class="git_failed")
    return res.stdout.decode("utf-8", errors="replace")


class GitStatus(Tool):
    name = "git_status"
    description = "Show git status of the workspace (branch, staged, modified, untracked files)."
    capability = Capability.GIT_READ

    class Input(ToolInput):
        pass

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        return ToolResult(True, (await _out(ctx, "status", "--short", "--branch")).strip() or "clean")


class GitDiff(Tool):
    name = "git_diff"
    description = ("Show a diff. against='baseline' (default in task workspaces) shows exactly the changes made in this task; "
                   "'HEAD' shows all uncommitted changes; any other value is treated as a git revision.")
    capability = Capability.GIT_READ

    class Input(ToolInput):
        against: str = "baseline"
        path: str | None = None
        stat_only: bool = False

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        if args.against == "baseline" and ctx.workspace.base_ref:
            diff = await ctx.services.workspaces.diff(ctx.workspace)
            if args.stat_only:
                lines = [f"{f.status} {f.path}" for f in diff.files]
                return ToolResult(True, "\n".join(lines) or "no changes", data=diff.stats)
            patch = diff.patch
            if args.path:
                chunks = patch.split("diff --git ")
                patch = "".join("diff --git " + c for c in chunks if c and (f" a/{args.path}" in c.split("\n")[0]))
            return ToolResult(True, patch or "no changes", data=diff.stats)
        rev = "HEAD" if args.against in ("baseline", "HEAD") else args.against
        cmd = ["diff", "--stat" if args.stat_only else "--no-color", rev]
        if args.path:
            _, rel = guard(ctx, args.path)
            cmd += ["--", rel]
        return ToolResult(True, (await _out(ctx, *cmd)).strip() or "no changes")


class GitLog(Tool):
    name = "git_log"
    description = "Show recent commits, optionally for a path."
    capability = Capability.GIT_READ

    class Input(ToolInput):
        limit: int = Field(20, ge=1, le=200)
        path: str | None = None

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        cmd = ["log", f"-n{args.limit}", "--date=short", "--pretty=format:%h %ad %an %s"]
        if args.path:
            _, rel = guard(ctx, args.path)
            cmd += ["--", rel]
        return ToolResult(True, (await _out(ctx, *cmd)).strip() or "no commits")


class GitShow(Tool):
    name = "git_show"
    description = "Show a commit (message and diff) or a file at a revision (rev:path)."
    capability = Capability.GIT_READ

    class Input(ToolInput):
        rev: str = Field(description="Commit-ish, or 'rev:path' for a file at a revision")

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        if args.rev.startswith("-"):
            raise ToolError("invalid revision", error_class="invalid_arguments")
        return ToolResult(True, (await _out(ctx, "show", "--no-color", "--stat", "--patch", args.rev))[:60_000])


class GitBlame(Tool):
    name = "git_blame"
    description = "Show who last changed each line in a range of a file."
    capability = Capability.GIT_READ

    class Input(ToolInput):
        path: str
        start: int = Field(1, ge=1)
        end: int = Field(80, ge=1)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        _, rel = guard(ctx, args.path)
        return ToolResult(True, (await _out(ctx, "blame", "--date=short", "-L", f"{args.start},{max(args.start, args.end)}", "--", rel)).strip())


class GitCommit(Tool):
    name = "git_commit"
    description = ("Create a local commit in the workspace (task branch in isolated workspaces). Never pushes. "
                   "Only use when the task asks for commits.")
    capability = Capability.GIT_WRITE
    side_effect = SideEffect.LOCAL
    read_only = False

    class Input(ToolInput):
        message: str = Field(min_length=3)
        paths: list[str] = Field(default_factory=list, description="Paths to commit; empty means all changes")

    def target(self, args: Input) -> str:
        return args.message[:80]

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        git = _git(ctx)
        excludes = [f":(exclude){p}" for p in ctx.workspace.meta.get("shared", [])]
        rels = [guard(ctx, p)[1] for p in args.paths] or ["."]
        await git.run("add", "-A", "--", *rels, *excludes)
        res = await git.run(*_IDENT, "commit", "--no-verify", "-q", "-m", args.message, check=False)
        if res.code != 0:
            raise ToolError(f"commit failed: {res.stderr or res.text}", error_class="git_failed")
        sha = (await git.out("rev-parse", "--short", "HEAD"))
        return ToolResult(True, f"committed {sha}: {args.message}", data={"commit": sha})


GIT_TOOLS: list[type[Tool]] = [GitStatus, GitDiff, GitLog, GitShow, GitBlame, GitCommit]
