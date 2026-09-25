"""Minimal async git runner used for internal workspace bookkeeping (not agent actions)."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from coremain.errors import WorkspaceError


@dataclass
class GitResult:
    code: int
    stdout: bytes
    stderr: str

    @property
    def text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace").strip()


def git_available() -> bool:
    return shutil.which("git") is not None


class Git:
    def __init__(self, work_tree: Path, git_dir: Path | None = None, *, index_file: Path | None = None):
        self.work_tree = work_tree
        self.git_dir = git_dir
        self.index_file = index_file

    def with_index(self, index_file: Path) -> Git:
        return Git(self.work_tree, self.git_dir, index_file=index_file)

    def _argv(self, args: tuple[str, ...]) -> list[str]:
        argv = ["git", "-c", "core.quotepath=off", "-c", "advice.detachedHead=false"]
        if self.git_dir is not None:
            argv += [f"--git-dir={self.git_dir}", f"--work-tree={self.work_tree}"]
        else:
            argv += ["-C", str(self.work_tree)]
        return argv + list(args)

    async def run(self, *args: str, check: bool = True, input: bytes | None = None, timeout: float = 300) -> GitResult:
        env = dict(os.environ)
        env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"})
        if self.index_file is not None:
            env["GIT_INDEX_FILE"] = str(self.index_file)
        proc = await asyncio.create_subprocess_exec(
            *self._argv(args),
            stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(input), timeout=timeout)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise WorkspaceError(f"git {' '.join(args[:3])} timed out") from exc
        result = GitResult(proc.returncode or 0, out, err.decode("utf-8", errors="replace").strip())
        if check and result.code != 0:
            raise WorkspaceError(f"git {' '.join(args[:4])} failed: {result.stderr or result.text}",
                                 details={"args": list(args[:8]), "code": result.code})
        return result

    async def out(self, *args: str, check: bool = True) -> str:
        return (await self.run(*args, check=check)).text


async def is_git_repo(path: Path) -> bool:
    if not git_available():
        return False
    res = await Git(path).run("rev-parse", "--is-inside-work-tree", check=False)
    return res.code == 0 and res.text == "true"


async def has_commits(path: Path) -> bool:
    res = await Git(path).run("rev-parse", "--verify", "-q", "HEAD", check=False)
    return res.code == 0
