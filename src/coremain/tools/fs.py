"""Filesystem tools: read, list, glob, search, write, edit, patch, delete.

Writes require that an existing file was read in this attempt and has not changed since
(read-before-write + staleness check), so concurrent human edits are detected instead of
overwritten. ``.git`` internals are never readable or writable through these tools.
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import re
import shutil
from pathlib import Path

from pydantic import Field

from coremain.errors import ToolError
from coremain.security.paths import PathViolation, resolve_within, sensitive_reason, to_relative
from coremain.security.policy import Capability, PolicyRequest
from coremain.tools.base import SideEffect, Tool, ToolContext, ToolInput, ToolResult
from coremain.tools.patching import PatchError, apply_hunks, parse_patch
from coremain.util.jsonutil import sha256_hex

IGNORED_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".cache",
        ".gradle",
        ".idea",
        "coverage",
        ".turbo",
        ".parcel-cache",
    }
)
MAX_READ_LINES = 2000


def guard(ctx: ToolContext, raw: str) -> tuple[Path, str]:
    path = resolve_within(ctx.workspace.path, raw)
    rel = to_relative(ctx.workspace.path, path)
    parts = Path(rel).parts
    if parts and parts[0] == ".git":
        raise PathViolation("the .git directory is managed by Core Main; use the git tools instead")
    return path, rel


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _check_fresh(ctx: ToolContext, rel: str, path: Path) -> None:
    if not path.exists():
        return
    current = sha256_hex(path.read_bytes())
    last = ctx.read_hashes.get(rel)
    if last is None:
        raise ToolError(
            f"read {rel} before modifying it (existing files must be read first)", error_class="read_required"
        )
    if last != current:
        raise ToolError(
            f"{rel} changed since you last read it (a human or another process modified it); read it again",
            error_class="stale_read",
        )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    tmp = path.with_name(f".{path.name}.core-tmp")
    tmp.write_bytes(data)
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def _numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{i:>6}\t{line[:2000]}" for i, line in enumerate(lines, start))


def _walk(root: Path, base: Path, extra_ignores: list[str]) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in IGNORED_DIRS and not any(fnmatch.fnmatch(d, p) for p in extra_ignores)
        )
        for name in sorted(filenames):
            out.append(Path(dirpath) / name)
    return out


class _PathInput(ToolInput):
    path: str = Field(description="Path relative to the workspace root")


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file from the workspace with line numbers. Use offset/limit for large files. "
        "You must read an existing file before editing it."
    )
    capability = Capability.FS_READ

    class Input(ToolInput):
        path: str = Field(description="Path relative to the workspace root")
        offset: int = Field(1, ge=1, description="1-based line to start from")
        limit: int = Field(400, ge=1, le=MAX_READ_LINES, description="Maximum number of lines")

    def target(self, args: Input) -> str:
        return args.path

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        path, rel = guard(ctx, args.path)
        if not path.exists():
            raise ToolError(f"{rel} does not exist", error_class="not_found")
        if path.is_dir():
            raise ToolError(f"{rel} is a directory; use list_dir", error_class="is_directory")
        data = await asyncio.to_thread(path.read_bytes)
        ctx.read_hashes[rel] = sha256_hex(data)
        ctx.files_read.add(rel)
        if _is_binary(data):
            return ToolResult(True, f"[binary file {rel}, {len(data)} bytes]")
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        chunk = lines[args.offset - 1 : args.offset - 1 + args.limit]
        header = f"{rel} ({len(lines)} lines)"
        if args.offset > 1 or args.offset - 1 + args.limit < len(lines):
            header += f", showing {args.offset}-{args.offset - 1 + len(chunk)}"
        return ToolResult(
            True, header + "\n" + _numbered(chunk, args.offset), data={"lines": len(lines), "path": rel}
        )


class ListDir(Tool):
    name = "list_dir"
    description = "List files and directories (ignores dependency/build directories)."
    capability = Capability.FS_READ

    class Input(ToolInput):
        path: str = "."
        depth: int = Field(2, ge=1, le=6)

    def target(self, args: Input) -> str:
        return args.path

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        base, rel = guard(ctx, args.path)
        if not base.is_dir():
            raise ToolError(f"{rel} is not a directory", error_class="not_a_directory")
        lines: list[str] = []

        def walk(d: Path, level: int) -> None:
            if level > args.depth or len(lines) > 600:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return
            for entry in entries:
                if entry.name in IGNORED_DIRS or entry.name == ".git":
                    continue
                indent = "  " * (level - 1)
                if entry.is_dir():
                    lines.append(f"{indent}{entry.name}/")
                    walk(entry, level + 1)
                else:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    lines.append(f"{indent}{entry.name} ({size} B)")

        await asyncio.to_thread(walk, base, 1)
        if len(lines) > 600:
            lines = [*lines[:600], "… (truncated)"]
        return ToolResult(True, f"{rel}/\n" + "\n".join(lines))


class GlobFiles(Tool):
    name = "glob_files"
    description = "Find files by glob pattern, e.g. '**/*.py' or 'src/**/test_*.ts'."
    capability = Capability.FS_READ

    class Input(ToolInput):
        pattern: str
        path: str = "."
        limit: int = Field(300, ge=1, le=2000)

    def target(self, args: Input) -> str:
        return args.path

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        base, _ = guard(ctx, args.path)
        files = await asyncio.to_thread(_walk, ctx.workspace.path, base, [])
        root = ctx.workspace.path.resolve()
        matches = []
        for f in files:
            rel = (
                f.resolve().relative_to(root).as_posix() if f.resolve().is_relative_to(root) else f.as_posix()
            )
            if fnmatch.fnmatch(rel, args.pattern) or fnmatch.fnmatch(f.name, args.pattern):
                matches.append(rel)
        shown = matches[: args.limit]
        more = f"\n… {len(matches) - len(shown)} more" if len(matches) > len(shown) else ""
        return ToolResult(
            True, "\n".join(shown) + more if shown else "no matches", data={"count": len(matches)}
        )


class SearchText(Tool):
    name = "search_text"
    description = "Search file contents (ripgrep when available). Returns path:line:text matches. Sensitive files are excluded."
    capability = Capability.FS_READ

    class Input(ToolInput):
        pattern: str = Field(description="Text or regular expression")
        path: str = "."
        regex: bool = False
        glob: str | None = Field(None, description="Restrict to files matching this glob, e.g. '*.py'")
        case_sensitive: bool = False
        max_results: int = Field(100, ge=1, le=1000)

    def target(self, args: Input) -> str:
        return args.path

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        base, _ = guard(ctx, args.path)
        root = ctx.workspace.path.resolve()
        rg = shutil.which("rg")
        results: list[str] = []
        if rg:
            argv = [rg, "-n", "--no-heading", "--color", "never", "--max-columns", "300", "--max-count", "50"]
            if not args.regex:
                argv.append("-F")
            if not args.case_sensitive:
                argv.append("-i")
            if args.glob:
                argv += ["--glob", args.glob]
            for pat in (".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*", ".git"):
                argv += ["--glob", f"!{pat}"]
            argv += ["--", args.pattern, str(base)]
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await proc.communicate()
            if proc.returncode not in (0, 1):
                raise ToolError(
                    f"search failed: {err.decode(errors='replace')[:300]}", error_class="search_failed"
                )
            for line in out.decode("utf-8", errors="replace").splitlines():
                path_part = line.split(":", 1)[0]
                try:
                    rel = Path(path_part).resolve().relative_to(root).as_posix()
                except ValueError:
                    rel = path_part
                results.append(rel + line[len(path_part) :])
                if len(results) >= args.max_results:
                    break
        else:
            flags = 0 if args.case_sensitive else re.IGNORECASE
            try:
                rx = re.compile(args.pattern if args.regex else re.escape(args.pattern), flags)
            except re.error as exc:
                raise ToolError(f"invalid regex: {exc}", error_class="invalid_arguments") from exc
            for f in await asyncio.to_thread(_walk, root, base, []):
                rel = f.resolve().relative_to(root).as_posix()
                if sensitive_reason(rel) or (
                    args.glob
                    and not fnmatch.fnmatch(f.name, args.glob)
                    and not fnmatch.fnmatch(rel, args.glob)
                ):
                    continue
                try:
                    data = f.read_bytes()
                except OSError:
                    continue
                if _is_binary(data):
                    continue
                for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        results.append(f"{rel}:{n}:{line[:300]}")
                        if len(results) >= args.max_results:
                            break
                if len(results) >= args.max_results:
                    break
        if not results:
            return ToolResult(True, "no matches", data={"count": 0})
        return ToolResult(True, "\n".join(results), data={"count": len(results)})


class WriteFile(Tool):
    name = "write_file"
    description = "Create a new file or fully replace an existing one (existing files must be read first)."
    capability = Capability.FS_WRITE
    side_effect = SideEffect.LOCAL
    read_only = False

    class Input(ToolInput):
        path: str
        content: str

    def target(self, args: Input) -> str:
        return args.path

    def summarize(self, args: Input) -> str:
        return f"write {args.path} ({len(args.content)} chars)"

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        path, rel = guard(ctx, args.path)
        if path.is_dir():
            raise ToolError(f"{rel} is a directory", error_class="is_directory")
        existed = path.exists()
        old = path.read_text(encoding="utf-8", errors="replace") if existed else ""
        _check_fresh(ctx, rel, path)
        data = args.content.encode("utf-8")
        await asyncio.to_thread(_atomic_write, path, data)
        ctx.read_hashes[rel] = sha256_hex(data)
        ctx.changed_paths.add(rel)
        added = removed = 0
        for line in difflib.unified_diff(old.splitlines(), args.content.splitlines(), lineterm="", n=0):
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        verb = "updated" if existed else "created"
        return ToolResult(
            True, f"{verb} {rel} (+{added} -{removed} lines)", data={"path": rel, "created": not existed}
        )


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must match exactly once (including whitespace) unless "
        "replace_all is true. Read the file first."
    )
    capability = Capability.FS_WRITE
    side_effect = SideEffect.LOCAL
    read_only = False

    class Input(ToolInput):
        path: str
        old_string: str = Field(min_length=1)
        new_string: str
        replace_all: bool = False

    def target(self, args: Input) -> str:
        return args.path

    def summarize(self, args: Input) -> str:
        return f"edit {args.path}"

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        path, rel = guard(ctx, args.path)
        if not path.exists():
            raise ToolError(f"{rel} does not exist; use write_file to create it", error_class="not_found")
        _check_fresh(ctx, rel, path)
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(args.old_string)
        if count == 0:
            lines = text.splitlines()
            probe = args.old_string.strip().splitlines()[0] if args.old_string.strip() else args.old_string
            close = difflib.get_close_matches(probe, [line.strip() for line in lines], n=3, cutoff=0.6)
            hint = (
                ("closest lines: " + " | ".join(close)) if close else "re-read the file to get the exact text"
            )
            raise ToolError(f"old_string not found in {rel}; {hint}", error_class="no_match")
        if count > 1 and not args.replace_all:
            raise ToolError(
                f"old_string occurs {count} times in {rel}; add surrounding context or set replace_all=true",
                error_class="ambiguous_match",
            )
        new_text = (
            text.replace(args.old_string, args.new_string)
            if args.replace_all
            else text.replace(args.old_string, args.new_string, 1)
        )
        data = new_text.encode("utf-8")
        await asyncio.to_thread(_atomic_write, path, data)
        ctx.read_hashes[rel] = sha256_hex(data)
        ctx.changed_paths.add(rel)
        idx = new_text.find(args.new_string)
        line_no = new_text.count("\n", 0, max(idx, 0)) + 1
        new_lines = new_text.splitlines()
        start = max(1, line_no - 3)
        snippet = _numbered(new_lines[start - 1 : line_no + args.new_string.count("\n") + 3], start)
        return ToolResult(
            True,
            f"edited {rel} ({count if args.replace_all else 1} replacement(s))\n{snippet}",
            data={"path": rel},
        )


class ApplyPatch(Tool):
    name = "apply_patch"
    description = (
        "Apply a multi-file patch atomically. Accepts a unified diff (---/+++/@@) or a '*** Begin Patch' envelope "
        "with '*** Add File:', '*** Update File:', '*** Delete File:' sections. All hunks must apply or nothing changes."
    )
    capability = Capability.FS_WRITE
    side_effect = SideEffect.LOCAL
    read_only = False

    class Input(ToolInput):
        patch: str = Field(min_length=1)

    def target(self, args: Input) -> str:
        try:
            return ",".join(p.path for p in parse_patch(args.patch))
        except PatchError:
            return "(unparseable patch)"

    def policy_request(self, ctx: ToolContext, args: Input) -> PolicyRequest | None:
        # Each file is checked individually inside run(); the request covers the first path for approvals.
        req = super().policy_request(ctx, args)
        if req is not None:
            req.target = self.target(args).split(",")[0]
        return req

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        try:
            patches = parse_patch(args.patch)
        except PatchError as exc:
            raise ToolError(str(exc), error_class="invalid_patch") from exc
        planned: list[tuple[Path, str, bytes | None]] = []
        for fp in patches:
            path, rel = guard(ctx, fp.path)
            cap = Capability.FS_DELETE if fp.op == "delete" else Capability.FS_WRITE
            decision = ctx.services.policy.evaluate(
                PolicyRequest(
                    capability=cap,
                    target=rel,
                    workspace_root=ctx.workspace.path,
                    workspace_isolated=ctx.workspace.isolated,
                    project_id=ctx.project_id,
                    session_id=ctx.session_id,
                    task_id=ctx.task_id,
                    tool=self.name,
                )
            )
            if decision.decision != "allow":
                raise ToolError(
                    f"patch touches {rel}: {decision.decision} by policy ({decision.reason})",
                    error_class="policy_denied",
                )
            if fp.op == "add":
                if path.exists():
                    raise ToolError(
                        f"cannot add {rel}: file already exists (use an update section)", error_class="exists"
                    )
                body = "\n".join(fp.content) + ("" if fp.no_newline_at_end else "\n")
                planned.append((path, rel, body.encode("utf-8")))
            elif fp.op == "delete":
                if not path.exists():
                    raise ToolError(f"cannot delete {rel}: file does not exist", error_class="not_found")
                _check_fresh(ctx, rel, path)
                planned.append((path, rel, None))
            else:
                if not path.exists():
                    raise ToolError(f"cannot update {rel}: file does not exist", error_class="not_found")
                _check_fresh(ctx, rel, path)
                try:
                    updated = apply_hunks(
                        path.read_text(encoding="utf-8", errors="replace"), fp.hunks, path=rel
                    )
                except PatchError as exc:
                    raise ToolError(str(exc), error_class="patch_failed") from exc
                planned.append((path, rel, updated.encode("utf-8")))
        for path, rel, data in planned:
            if data is None:
                path.unlink()
                ctx.read_hashes.pop(rel, None)
            else:
                await asyncio.to_thread(_atomic_write, path, data)
                ctx.read_hashes[rel] = sha256_hex(data)
            ctx.changed_paths.add(rel)
        summary = ", ".join(f"{'deleted' if d is None else 'wrote'} {r}" for _, r, d in planned)
        return ToolResult(True, f"patch applied: {summary}", data={"files": [r for _, r, _ in planned]})


class DeleteFile(Tool):
    name = "delete_file"
    description = "Delete a file in the workspace (must have been read first)."
    capability = Capability.FS_DELETE
    side_effect = SideEffect.LOCAL
    read_only = False
    Input = _PathInput

    def target(self, args: _PathInput) -> str:
        return args.path

    async def run(self, ctx: ToolContext, args: _PathInput) -> ToolResult:
        path, rel = guard(ctx, args.path)
        if not path.is_file():
            raise ToolError(f"{rel} is not an existing file", error_class="not_found")
        _check_fresh(ctx, rel, path)
        path.unlink()
        ctx.read_hashes.pop(rel, None)
        ctx.changed_paths.add(rel)
        return ToolResult(True, f"deleted {rel}")


FS_TOOLS: list[type[Tool]] = [
    ReadFile,
    ListDir,
    GlobFiles,
    SearchText,
    WriteFile,
    EditFile,
    ApplyPatch,
    DeleteFile,
]
