"""Objective post-conditions for evaluation scenarios, evaluated against the canonical checkout
and the sandbox runtime's durable records (never against model claims)."""

from __future__ import annotations

import asyncio
import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coremain.evals.scenarios import expand
from coremain.verify.evidence import LEVELS

if TYPE_CHECKING:
    from coremain.domain.models import Task
    from coremain.runtime.app import CoreRuntime


@dataclass
class CheckContext:
    rt: CoreRuntime
    task: Task
    repo: Path
    initial_head: str
    initial_status: str
    env: dict[str, str]

    def final_message(self) -> str:
        if not self.task.session_id:
            return ""
        msgs = [m for m in self.rt.sessions.messages(self.task.session_id) if m.task_id == self.task.id]
        return msgs[-1].content if msgs else ""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False).stdout


def changed_files(repo: Path, initial_head: str) -> list[str]:
    from coremain.workspaces.manager import is_generated

    tracked = _git(repo, "diff", "--name-only", initial_head).splitlines()
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted({p for p in (*tracked, *untracked) if p and not (p in untracked and is_generated(p))})


def _read(ctx: CheckContext, rel: str) -> str | None:
    path = ctx.repo / rel
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


async def _command(ctx: CheckContext, spec: Any) -> tuple[bool, str]:
    if isinstance(spec, str):
        spec = {"run": spec}
    cmd = expand(str(spec["run"]))
    expect = int(spec.get("expect_exit", 0))
    proc = await asyncio.create_subprocess_shell(
        cmd,
        cwd=ctx.repo,
        env={**ctx.env, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=float(spec.get("timeout_s", 300)))
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False, f"`{cmd}` timed out"
    text = out.decode("utf-8", errors="replace")
    ok = proc.returncode == expect
    if ok and spec.get("expect_output") and spec["expect_output"] not in text:
        return False, f"`{cmd}` output lacks {spec['expect_output']!r}"
    tail = text.strip().splitlines()[-3:]
    return ok, f"`{cmd}` exited {proc.returncode} (expected {expect}): {' | '.join(tail)[:300]}"


async def evaluate(check: dict[str, Any], ctx: CheckContext) -> dict[str, Any]:
    kind, arg = next(iter(check.items()))
    ok, detail = False, ""
    task = ctx.task
    if kind == "status":
        allowed = [arg] if isinstance(arg, str) else list(arg)
        ok, detail = task.status.value in allowed, f"task {task.status.value} ({task.status_reason or ''})"
    elif kind in {"file_contains", "file_not_contains"}:
        content = _read(ctx, arg["path"])
        texts = arg["text"] if isinstance(arg["text"], list) else [arg["text"]]
        present = [t for t in texts if content is not None and t in content]
        if kind == "file_contains":
            ok = content is not None and len(present) == len(texts)
            detail = (
                "missing file"
                if content is None
                else f"found {len(present)}/{len(texts)} expected snippet(s)"
            )
        else:
            ok = content is not None and not present
            detail = (
                "missing file"
                if content is None
                else (f"still contains {present[0]!r}" if present else "absent")
            )
    elif kind == "file_matches":
        content = _read(ctx, arg["path"]) or ""
        ok = re.search(arg["regex"], content, re.M) is not None
        detail = f"/{arg['regex']}/ {'matched' if ok else 'not matched'} in {arg['path']}"
    elif kind == "file_exists":
        ok, detail = (ctx.repo / arg).exists(), arg
    elif kind == "file_absent":
        ok, detail = not (ctx.repo / arg).exists(), arg
    elif kind == "command":
        ok, detail = await _command(ctx, arg)
    elif kind == "evidence":
        evidence = ctx.rt.evidence.for_task(task.id)
        match = [e for e in evidence if e.kind == arg["kind"] and e.status == arg.get("status", "pass")]
        ok, detail = bool(match), f"{len(match)} {arg['kind']}={arg.get('status', 'pass')} record(s)"
    elif kind == "min_level":
        level = task.evidence_level or "none"
        ok, detail = LEVELS.index(level) >= LEVELS.index(arg), f"evidence level {level}"
    elif kind == "canonical_unchanged":
        head = _git(ctx.repo, "rev-parse", "HEAD").strip()
        status = _git(ctx.repo, "status", "--porcelain")
        ok = head == ctx.initial_head and status == ctx.initial_status
        detail = (
            "unchanged" if ok else f"changed: {', '.join(changed_files(ctx.repo, ctx.initial_head))[:200]}"
        )
    elif kind == "answer_contains":
        message = ctx.final_message().lower()
        needles = [arg] if isinstance(arg, str) else list(arg)
        missing = [n for n in needles if n.lower() not in message]
        ok, detail = not missing, f"missing {missing}" if missing else "all present"
    elif kind == "review_verdict":
        verdicts = [r["verdict"] for r in ctx.rt.reviews.reviews(task.id) if r["reviewer"] != "deterministic"]
        ok, detail = arg in verdicts, f"verdicts {verdicts}"
    elif kind == "findings_min":
        n = int(ctx.rt.db.scalar("SELECT COUNT(*) FROM findings WHERE task_id = ?", (task.id,)) or 0)
        ok, detail = n >= int(arg), f"{n} finding(s)"
    elif kind == "changed_only":
        patterns = [arg] if isinstance(arg, str) else list(arg)
        files = changed_files(ctx.repo, ctx.initial_head)
        outside = [f for f in files if not any(fnmatch.fnmatch(f, p) for p in patterns)]
        ok, detail = not outside, f"outside allowed paths: {outside}" if outside else f"changed {files}"
    elif kind == "max_model_calls":
        n = int(ctx.rt.db.scalar("SELECT COUNT(*) FROM model_calls WHERE task_id = ?", (task.id,)) or 0)
        ok, detail = n <= int(arg), f"{n} model call(s)"
    elif kind == "model_retries_min":
        n = int(
            ctx.rt.db.scalar(
                "SELECT COALESCE(SUM(retries), 0) FROM model_calls WHERE task_id = ?", (task.id,)
            )
            or 0
        )
        ok, detail = n >= int(arg), f"{n} retr(y/ies)"
    elif kind == "repo_not_contains":
        hits = [
            str(p.relative_to(ctx.repo))
            for p in ctx.repo.rglob("*")
            if p.is_file()
            and ".git" not in p.parts
            and arg in (p.read_text(encoding="utf-8", errors="ignore"))
        ]
        ok, detail = not hits, f"found in {hits}" if hits else "absent"
    return {"check": kind, "arg": arg, "ok": ok, "detail": detail}
