"""Workspace manager.

Kinds:
* ``worktree`` – a git worktree of the project on branch ``core/<task>``. The user's
  uncommitted tracked changes and untracked files are carried over, then recorded in a
  *baseline commit*, so the agent's changes are exactly ``diff(baseline, current)``.
* ``copy`` – for non-git projects: a filtered copy with a private repository and baseline.
* ``canonical`` – the user's working tree. Used read-only, or in explicitly authorized direct
  mode, where a shadow repository (outside the project) records the baseline.

Applying an isolated workspace to the canonical tree is all-or-nothing: every changed file is
checked against the baseline; files the human changed meanwhile are 3-way merged with
``git merge-file`` and any conflict aborts the whole apply without writing anything.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coremain.config.schema import WorkspaceConfig
from coremain.domain.models import Project
from coremain.errors import NotFoundError, WorkspaceConflictError, WorkspaceError
from coremain.events import EventLog
from coremain.paths import CorePaths
from coremain.store.artifacts import ArtifactStore
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id, short_id
from coremain.util.jsonutil import dumps, loads, sha256_hex
from coremain.workspaces.git import Git, has_commits, is_git_repo

COPY_IGNORES = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "target", ".next", ".nuxt", ".tox", ".cache", ".gradle", ".idea", ".DS_Store", "coverage", ".coverage",
})
SHADOW_EXCLUDES = "\n".join(sorted(COPY_IGNORES | {".core/cache", ".core/tmp"})) + "\n"
_IDENT = ("-c", "user.name=Core Main", "-c", "user.email=core-main@localhost", "-c", "commit.gpgsign=false")


@dataclass
class Workspace:
    id: str
    project_id: str
    kind: str
    path: Path
    branch: str | None
    base_ref: str | None
    status: str
    owner_task_id: str | None
    owner_attempt_id: str | None
    meta: dict[str, Any]
    created_at: float

    @property
    def isolated(self) -> bool:
        return self.kind in {"worktree", "copy"}

    @classmethod
    def from_row(cls, r: Any) -> Workspace:
        return cls(r["id"], r["project_id"], r["kind"], Path(r["path"]), r["branch"], r["base_ref"], r["status"],
                   r["owner_task_id"], r["owner_attempt_id"], loads(r["meta_json"], {}), r["created_at"])

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["path"] = str(self.path)
        return d


@dataclass
class FileChange:
    status: str  # A | M | D | T
    path: str


@dataclass
class WorkspaceDiff:
    files: list[FileChange]
    patch: str
    diff_hash: str
    fingerprint: str
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.files

    def paths(self) -> list[str]:
        return [f.path for f in self.files]


@dataclass
class ApplyResult:
    applied: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.conflicts

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GcAction:
    workspace_id: str | None
    path: str
    action: str
    reason: str


def _read_bytes(path: Path) -> bytes | None:
    try:
        if path.is_symlink():
            return os.readlink(path).encode()
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def _is_text(*blobs: bytes | None) -> bool:
    return all(b is None or b"\x00" not in b[:8192] for b in blobs)


class WorkspaceManager:
    def __init__(self, db: Database, events: EventLog, artifacts: ArtifactStore, paths: CorePaths, clock: Clock,
                 config: WorkspaceConfig):
        self.db = db
        self.events = events
        self.artifacts = artifacts
        self.paths = paths
        self.clock = clock
        self.config = config
        self._locks: dict[str, asyncio.Lock] = {}

    # ----------------------------------------------------------------- records
    def _insert(self, *, project_id: str, kind: str, path: Path, branch: str | None, base_ref: str | None, status: str,
                task_id: str | None, attempt_id: str | None, meta: dict[str, Any], ws_id: str | None = None) -> Workspace:
        now = self.clock.now()
        ws_id = ws_id or new_id("ws", now=now)
        self.db.execute(
            "INSERT INTO workspaces(id, project_id, kind, path, branch, base_ref, status, owner_task_id, owner_attempt_id, meta_json, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ws_id, project_id, kind, str(path), branch, base_ref, status, task_id, attempt_id, dumps(meta), now, now),
        )
        return self.get(ws_id)

    def get(self, ws_id: str) -> Workspace:
        row = self.db.one("SELECT * FROM workspaces WHERE id = ?", (ws_id,))
        if row is None:
            raise NotFoundError(f"workspace {ws_id} not found")
        return Workspace.from_row(row)

    def list(self, project_id: str | None = None, *, statuses: tuple[str, ...] | None = None) -> list[Workspace]:
        sql = "SELECT * FROM workspaces WHERE 1=1"
        params: list[Any] = []
        if project_id:
            sql += " AND project_id = ?"
            params.append(project_id)
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params.extend(statuses)
        return [Workspace.from_row(r) for r in self.db.query(sql + " ORDER BY created_at DESC", params)]

    def set_status(self, ws: Workspace, status: str, **meta: Any) -> Workspace:
        merged = {**ws.meta, **meta}
        self.db.execute("UPDATE workspaces SET status = ?, meta_json = ?, updated_at = ? WHERE id = ?",
                        (status, dumps(merged), self.clock.now(), ws.id))
        self.events.emit("workspace.status", project_id=ws.project_id, task_id=ws.owner_task_id,
                         data={"workspace_id": ws.id, "status": status, **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}})
        return self.get(ws.id)

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    # -------------------------------------------------------------- git views
    def _shadow_dir(self, project_id: str) -> Path:
        return self.paths.snapshots_dir / f"{project_id}.git"

    def git_for(self, ws: Workspace) -> Git:
        if ws.kind == "canonical":
            return Git(ws.path, self._shadow_dir(ws.project_id))
        return Git(ws.path)

    def _index_file(self, ws: Workspace) -> Path:
        d = self.paths.data_dir / "indexes"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{ws.id}.index"

    def _excludes(self, ws: Workspace) -> list[str]:
        return [f":(exclude){p}" for p in ws.meta.get("shared", [])]

    async def _ensure_shadow(self, project_id: str, root: Path) -> Git:
        shadow = self._shadow_dir(project_id)
        if not (shadow / "HEAD").exists():
            shadow.mkdir(parents=True, exist_ok=True)
            await Git(root).run("init", "-q", "--bare", str(shadow))
        info = shadow / "info"
        info.mkdir(exist_ok=True)
        (info / "exclude").write_text(SHADOW_EXCLUDES + ".git\n", encoding="utf-8")
        return Git(root, shadow)

    # --------------------------------------------------------------- creation
    def canonical(self, project: Project) -> Workspace:
        row = self.db.one("SELECT * FROM workspaces WHERE project_id = ? AND kind = 'canonical' AND base_ref IS NULL "
                          "AND status = 'active' ORDER BY created_at LIMIT 1", (project.id,))
        if row is not None:
            return Workspace.from_row(row)
        return self._insert(project_id=project.id, kind="canonical", path=Path(project.root_path), branch=None, base_ref=None,
                            status="active", task_id=None, attempt_id=None, meta={"read_only_view": True})

    async def create_direct(self, project: Project, task_id: str, attempt_id: str | None) -> Workspace:
        """Direct mode: the canonical tree itself, with a shadow-repo baseline for diff/undo."""
        root = Path(project.root_path)
        git = await self._ensure_shadow(project.id, root)
        ws = self._insert(project_id=project.id, kind="canonical", path=root, branch=None, base_ref=None, status="active",
                          task_id=task_id, attempt_id=attempt_id, meta={"direct": True})
        base = await self._commit_state(git.with_index(self._index_file(ws)), f"core: direct-mode baseline for {task_id}", [])
        self.db.execute("UPDATE workspaces SET base_ref = ? WHERE id = ?", (base, ws.id))
        self.events.emit("workspace.created", project_id=project.id, task_id=task_id,
                         data={"workspace_id": ws.id, "kind": "canonical", "direct": True})
        return self.get(ws.id)

    async def create_isolated(self, project: Project, task_id: str, attempt_id: str | None, *,
                              from_commit: str | None = None) -> Workspace:
        root = Path(project.root_path)
        ws_id = new_id("ws")
        dest = self.paths.workspaces_dir / project.id / short_id(ws_id, 10).split("_", 1)[1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        git_ok = await is_git_repo(root) and await has_commits(root)
        shared: list[str] = []
        if git_ok:
            branch = f"core/{short_id(task_id)}-{short_id(ws_id, 6).split('_', 1)[1]}"
            await Git(root).run("worktree", "add", "-q", "-b", branch, str(dest), from_commit or "HEAD")
            kind = "worktree"
            if from_commit is None:
                await self._carry_uncommitted(root, dest)
        else:
            branch = None
            kind = "copy"
            await asyncio.to_thread(self._copy_tree, root, dest)
            await Git(dest).run("init", "-q")
        for rel in self.config.share_paths:
            src = root / rel
            target = dest / rel
            if src.exists() and not target.exists() and not os.path.islink(target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(src, target_is_directory=src.is_dir())
                shared.append(rel)
        ws = self._insert(project_id=project.id, kind=kind, path=dest, branch=branch, base_ref=None, status="active",
                          task_id=task_id, attempt_id=attempt_id, meta={"shared": shared, "source": str(root)}, ws_id=ws_id)
        git = Git(dest)
        base = await self._commit_state(git, f"core: baseline for {task_id}", self._excludes(ws), commit_to_head=True)
        self.db.execute("UPDATE workspaces SET base_ref = ? WHERE id = ?", (base, ws.id))
        self.events.emit("workspace.created", project_id=project.id, task_id=task_id,
                         data={"workspace_id": ws.id, "kind": kind, "path": str(dest), "branch": branch, "shared": shared})
        return self.get(ws.id)

    async def _carry_uncommitted(self, root: Path, dest: Path) -> None:
        src_git = Git(root)
        patch = (await src_git.run("diff", "HEAD", "--binary")).stdout
        if patch.strip():
            res = await Git(dest).run("apply", "--binary", "--whitespace=nowarn", input=patch, check=False)
            if res.code != 0:
                names = (await src_git.run("diff", "HEAD", "--name-only", "-z")).stdout.split(b"\x00")
                for raw in names:
                    if raw:
                        self._copy_file(root, dest, raw.decode())
        untracked = (await src_git.run("ls-files", "--others", "--exclude-standard", "-z")).stdout.split(b"\x00")
        for raw in untracked:
            if raw:
                self._copy_file(root, dest, raw.decode())

    @staticmethod
    def _copy_file(root: Path, dest: Path, rel: str) -> None:
        src = root / rel
        target = dest / rel
        if src.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.readlink(src))
        elif src.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        elif not src.exists() and (target.exists() or target.is_symlink()):
            target.unlink()

    def _copy_tree(self, root: Path, dest: Path) -> None:
        limit = self.config.copy_max_mb * 1024 * 1024
        total = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in COPY_IGNORES and d not in self.config.share_paths]
            rel_dir = Path(dirpath).relative_to(root)
            for name in filenames:
                src = Path(dirpath) / name
                try:
                    total += src.lstat().st_size
                except OSError:
                    continue
                if total > limit:
                    shutil.rmtree(dest, ignore_errors=True)
                    raise WorkspaceError(
                        f"project exceeds workspace.copy_max_mb ({self.config.copy_max_mb} MB) for copy isolation",
                        hint="Initialize a git repository (worktree isolation is cheap) or use direct mode (--direct).",
                    )
                self._copy_file(root, dest, str(rel_dir / name))
        dest.mkdir(parents=True, exist_ok=True)

    async def _commit_state(self, git: Git, message: str, excludes: list[str], *, commit_to_head: bool = False) -> str:
        await git.run("add", "-A", "--", ".", *excludes)
        if commit_to_head:
            await git.run(*_IDENT, "commit", "--no-verify", "--allow-empty", "-q", "-m", message)
            return await git.out("rev-parse", "HEAD")
        tree = await git.out("write-tree")
        return await git.out(*_IDENT, "commit-tree", tree, "-m", message)

    # ------------------------------------------------------ state inspection
    async def fingerprint(self, ws: Workspace) -> str:
        """Content-addressed tree id of the workspace (excluding ignored and shared paths)."""
        if ws.kind == "canonical" and ws.base_ref is None:
            git = await self._ensure_shadow(ws.project_id, ws.path)
        else:
            git = self.git_for(ws)
        idx = git.with_index(self._index_file(ws))
        await idx.run("add", "-A", "--", ".", *self._excludes(ws))
        return await idx.out("write-tree")

    async def diff(self, ws: Workspace) -> WorkspaceDiff:
        if ws.base_ref is None:
            fp = await self.fingerprint(ws)
            return WorkspaceDiff([], "", sha256_hex(""), fp)
        git = self.git_for(ws)
        tree = await self.fingerprint(ws)
        names = (await git.run("diff", "--no-renames", "--name-status", "-z", ws.base_ref, tree)).stdout.split(b"\x00")
        files: list[FileChange] = []
        it = iter(n.decode() for n in names if n)
        for status_code in it:
            path = next(it, "")
            files.append(FileChange(status_code[:1], path))
        patch = (await git.run("diff", "--no-renames", "--binary", ws.base_ref, tree)).stdout.decode("utf-8", errors="replace")
        numstat = await git.out("diff", "--no-renames", "--numstat", ws.base_ref, tree)
        added = removed = 0
        for line in numstat.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                added += int(parts[0])
                removed += int(parts[1])
        return WorkspaceDiff(files, patch, sha256_hex(patch), tree, {"files": len(files), "added": added, "removed": removed})

    async def baseline_blob(self, ws: Workspace, rel: str) -> bytes | None:
        if ws.base_ref is None:
            return None
        res = await self.git_for(ws).run("show", f"{ws.base_ref}:{rel}", check=False)
        return res.stdout if res.code == 0 else None

    # ----------------------------------------------------------- reconciliation
    async def apply(self, src: Workspace, dst_root: Path, *, dry_run: bool = False) -> ApplyResult:
        """Apply ``src``'s changes (relative to its baseline) onto ``dst_root``; all-or-nothing."""
        if not src.isolated:
            raise WorkspaceError("only isolated workspaces can be applied")
        diff = await self.diff(src)
        result = ApplyResult()
        plan: list[tuple[str, bytes | None, int | None]] = []
        for change in diff.files:
            base = await self.baseline_blob(src, change.path)
            new = None if change.status == "D" else _read_bytes(src.path / change.path)
            dst_file = dst_root / change.path
            cur = _read_bytes(dst_file)
            mode = None
            if new is not None and (src.path / change.path).exists() and not (src.path / change.path).is_symlink():
                mode = stat.S_IMODE((src.path / change.path).stat().st_mode)
            if cur == new:
                result.already.append(change.path)
            elif cur == base:
                plan.append((change.path, new, mode))
                result.applied.append(change.path)
            elif base is not None and cur is not None and new is not None and _is_text(base, cur, new):
                merged, clean = await self._merge3(cur, base, new)
                if clean:
                    plan.append((change.path, merged, mode))
                    result.merged.append(change.path)
                else:
                    result.conflicts.append({"path": change.path, "reason": "concurrent edits overlap (3-way merge conflict)"})
            else:
                why = "file was created concurrently" if base is None else "file changed concurrently and cannot be merged"
                result.conflicts.append({"path": change.path, "reason": why})
        if result.conflicts or dry_run:
            return result
        for rel, content, mode in plan:
            target = dst_root / rel
            if content is None:
                if target.exists() or target.is_symlink():
                    target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.core-tmp")
            tmp.write_bytes(content)
            if mode is not None:
                os.chmod(tmp, mode)
            os.replace(tmp, target)
        return result

    async def _merge3(self, current: bytes, base: bytes, new: bytes) -> tuple[bytes, bool]:
        with tempfile.TemporaryDirectory(prefix="core-merge-") as tmpdir:
            t = Path(tmpdir)
            (t / "current").write_bytes(current)
            (t / "base").write_bytes(base)
            (t / "new").write_bytes(new)
            res = await Git(t).run("merge-file", "-p", "-L", "canonical", "-L", "baseline", "-L", "task",
                                   str(t / "current"), str(t / "base"), str(t / "new"), check=False)
            return res.stdout, res.code == 0

    async def apply_to_canonical(self, src: Workspace, project: Project) -> ApplyResult:
        async with self.lock(f"canonical:{project.id}"):
            result = await self.apply(src, Path(project.root_path))
        if result.ok:
            self.set_status(src, "applied", applied_files=len(result.applied) + len(result.merged))
            self.events.emit("workspace.applied", project_id=project.id, task_id=src.owner_task_id, data=result.to_dict())
        else:
            self.events.emit("workspace.conflict", project_id=project.id, task_id=src.owner_task_id, level="warning",
                             data=result.to_dict())
            raise WorkspaceConflictError(
                f"{len(result.conflicts)} file(s) changed in the canonical workspace conflict with the task's changes",
                hint="Inspect with `core task diff`, resolve manually, then `core task apply` again; the task workspace is preserved.",
                details=result.to_dict(),
            )
        return result

    # --------------------------------------------------------------- checkpoints
    async def checkpoint(self, ws: Workspace, label: str) -> str:
        git = self.git_for(ws)
        tree = await self.fingerprint(ws)
        parents = ["-p", ws.base_ref] if ws.base_ref else []
        commit = await git.out(*_IDENT, "commit-tree", tree, *parents, "-m", label)
        await git.run("update-ref", f"refs/core/checkpoints/{ws.id}-{int(self.clock.now())}", commit)
        return commit

    # ------------------------------------------------------------------ cleanup
    async def remove(self, ws: Workspace, *, reason: str) -> None:
        if not ws.isolated:
            raise WorkspaceError("refusing to remove a canonical workspace")
        source = Path(ws.meta.get("source", ""))
        if ws.kind == "worktree" and source.exists():
            await Git(source).run("worktree", "remove", "--force", str(ws.path), check=False)
            if ws.branch:
                await Git(source).run("branch", "-D", ws.branch, check=False)
            await Git(source).run("worktree", "prune", check=False)
        if ws.path.exists():
            await asyncio.to_thread(shutil.rmtree, ws.path, True)
        self._index_file(ws).unlink(missing_ok=True)
        self.set_status(ws, "deleted", removed_reason=reason)

    def gc_plan(self, *, task_status: dict[str, str], force_orphans: bool = False) -> list[GcAction]:
        """Decide what can be deleted. Never touches active or recoverable work."""
        actions: list[GcAction] = []
        now = self.clock.now()
        keep_s = self.config.keep_days * 86400
        known_paths: set[str] = set()
        for ws in self.list():
            known_paths.add(str(ws.path))
            if not ws.isolated or ws.status == "deleted":
                continue
            owner_status = task_status.get(ws.owner_task_id or "", "unknown")
            age = now - ws.created_at
            if ws.status == "active":
                if owner_status in {"completed", "cancelled", "failed"} and age > keep_s:
                    actions.append(GcAction(ws.id, str(ws.path), "delete", f"owner task {owner_status}; stale active workspace"))
                continue
            if ws.status in {"applied", "discarded", "released"}:
                if age > keep_s or ws.status == "discarded":
                    actions.append(GcAction(ws.id, str(ws.path), "delete", f"workspace {ws.status}"))
                continue
            if ws.status == "preserved":
                if owner_status in {"completed", "cancelled"} and age > keep_s:
                    actions.append(GcAction(ws.id, str(ws.path), "delete", f"preserved workspace of {owner_status} task past retention"))
                else:
                    actions.append(GcAction(ws.id, str(ws.path), "keep", f"preserved for recovery (task {owner_status})"))
        root = self.paths.workspaces_dir
        if root.exists():
            for project_dir in root.iterdir():
                if not project_dir.is_dir():
                    continue
                for child in project_dir.iterdir():
                    if str(child) not in known_paths:
                        actions.append(GcAction(None, str(child), "delete" if force_orphans else "report",
                                                "directory has no workspace record (orphan)"))
        return actions

    async def gc(self, *, task_status: dict[str, str], dry_run: bool = True, force_orphans: bool = False) -> list[GcAction]:
        actions = self.gc_plan(task_status=task_status, force_orphans=force_orphans)
        if dry_run:
            return actions
        for action in actions:
            if action.action != "delete":
                continue
            if action.workspace_id:
                await self.remove(self.get(action.workspace_id), reason=action.reason)
            else:
                await asyncio.to_thread(shutil.rmtree, action.path, True)
        return actions
