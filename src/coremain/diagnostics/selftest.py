"""`core selftest`: verify Core Main itself, end to end, in a disposable installation.

Everything runs in a temporary CORE_HOME with a deterministic scripted provider and throwaway
git repositories, so the user's projects, history and credentials are never touched. The
crash check really SIGKILLs a child process mid-task and verifies recovery and resumption.
``live=True`` additionally performs one minimal model call with the user's real configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import tomli_w

from coremain.paths import resolve_core_paths

Progress = Callable[[str, bool, str], None]
_KEEP_ENV = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR")
_IDENT = [
    "-c",
    "user.name=Core Main Selftest",
    "-c",
    "user.email=selftest@localhost",
    "-c",
    "commit.gpgsign=false",
]

CALC = {
    "calc.py": "def add(a, b):\n    return a - b\n",
    "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    "pyproject.toml": "[tool.pytest.ini_options]\npythonpath = ['.']\n",
}


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"tool_calls": [{"name": name, "arguments": args}]}


FIX = [
    _call("read_file", path="calc.py"),
    _call("edit_file", path="calc.py", old_string="return a - b", new_string="return a + b"),
    _call("run_tests"),
    _call("submit_result", summary="add() now adds", files_changed=["calc.py"], confidence="high"),
]
APPROVE = [_call("submit_review", verdict="approve", summary="correct minimal fix")]
LIAR = [_call("submit_result", summary="All tests pass now", files_changed=["calc.py"], confidence="high")]
SLOW = [{"delay_s": 60, **_call("read_file", path="calc.py")}, *FIX[1:]]

# Child process for the crash check: starts a task whose first model call blocks, then gets SIGKILLed.
_CRASH_CHILD = """
import asyncio, os, sys
from pathlib import Path
from coremain.paths import resolve_core_paths
from coremain.runtime.app import CoreRuntime

async def main():
    env = dict(os.environ)
    rt = CoreRuntime.open(project_root=Path(sys.argv[1]), paths=resolve_core_paths(env), env=env)
    await rt.start()
    task = await rt.submit("Fix add() in calc.py so it adds", mode="direct")
    print(task.id, flush=True)
    await rt.run_task(task.id)

asyncio.run(main())
"""


class _Env:
    def __init__(self, base: Path, user_env: Mapping[str, str]):
        self.base = base
        home = base / "home"
        home.mkdir(parents=True, exist_ok=True)
        self.env = {k: v for k, v in user_env.items() if k in _KEEP_ENV}
        self.env.update({"HOME": str(home), "CORE_HOME": str(base / "core")})
        self.paths = resolve_core_paths(self.env)
        self.paths.ensure()
        self._n = 0

    def repo(self, files: Mapping[str, str]) -> Path:
        self._n += 1
        repo = self.base / f"repo{self._n}"
        repo.mkdir()
        for rel, content in files.items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(content, encoding="utf-8")
        for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "init"]):
            subprocess.run(["git", *_IDENT, *args], cwd=repo, check=True, capture_output=True)
        return repo

    def script(self, roles: dict[str, Any]) -> None:
        self._n += 1
        path = self.base / f"script{self._n}.json"
        path.write_text(json.dumps({"name": "selftest", "roles": roles}), encoding="utf-8")
        cfg = {
            "providers": {"selftest": {"kind": "scripted", "script": str(path)}},
            "models": {
                "scripted": {
                    "provider": "selftest",
                    "id": "scripted",
                    "context_window": 64000,
                    "max_output_tokens": 4096,
                }
            },
            "verification": {"commands": {"test": f"{sys.executable} -m pytest -q -p no:cacheprovider"}},
        }
        self.paths.config_file.write_text(tomli_w.dumps(cfg), encoding="utf-8")

    def runtime(self, project: Path | None = None, **kw: Any) -> Any:
        from coremain.runtime.app import CoreRuntime

        return CoreRuntime.open(project_root=project, paths=self.paths, env=self.env, **kw)


def _git_status(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False
    ).stdout.strip()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _storage(t: _Env) -> str:
    from coremain.store.migrate import status

    rt = t.runtime()
    try:
        problems = rt.db.integrity_check(full=True)
        _expect(not problems, f"integrity problems: {problems}")
        st = status(rt.db)
        _expect(st.current == st.latest and not st.checksum_mismatches, f"schema {st.current}/{st.latest}")
        return f"schema v{st.current}, WAL, integrity ok"
    finally:
        await rt.close()


async def _policy(t: _Env) -> str:
    from coremain.security.commands import analyze_command
    from coremain.security.policy import Capability, PolicyRequest

    ws = t.base / "ws"
    ws.mkdir(exist_ok=True)
    rt = t.runtime(overrides={"permissions": {"network": {"offline": True}}})
    try:
        cases = [
            (
                PolicyRequest(capability=Capability.FS_WRITE, target="/etc/passwd", workspace_root=ws),
                {"deny"},
                "write outside workspace",
            ),
            (
                PolicyRequest(
                    capability=Capability.EXEC,
                    target="rm -rf /",
                    workspace_root=ws,
                    analysis=analyze_command("rm -rf /", workspace=ws),
                ),
                {"deny"},
                "destructive command",
            ),
            (
                PolicyRequest(
                    capability=Capability.NET_HTTP, target="https://example.com", workspace_root=ws
                ),
                {"deny"},
                "network while offline",
            ),
            (
                PolicyRequest(
                    capability=Capability.GIT_REMOTE, target="git push origin main", workspace_root=ws
                ),
                {"deny", "ask"},
                "git push",
            ),
            (
                PolicyRequest(capability=Capability.FS_READ, target=str(ws / "notes.txt"), workspace_root=ws),
                {"allow"},
                "read in workspace",
            ),
        ]
        for req, allowed, label in cases:
            decision = rt.policy.evaluate(req)
            _expect(decision.decision in allowed, f"{label}: got {decision.decision} ({decision.reason})")
        return f"{len(cases)} invariants hold (outside-workspace writes, destructive commands, offline network, remote git)"
    finally:
        await rt.close()


async def _secrets(t: _Env) -> str:
    from coremain.security.env import build_subprocess_env
    from coremain.security.redact import Redactor

    secret = "Zq8vN2kLm4pR7tW1xY5bC9dF3gH6jK0s"  # noqa: S105 - synthetic value the detectors must catch
    redacted = Redactor().redact(f'api_key = "{secret}"')
    _expect(secret not in redacted, "a quoted api_key survived redaction")
    env = build_subprocess_env(
        {"PATH": "/usr/bin", "OPENAI_API_KEY": secret, "AWS_SECRET_ACCESS_KEY": secret}, passthrough=[]
    )
    _expect(secret not in env.values(), "a provider key leaked into a child environment")
    return "secrets redacted from text and stripped from child environments"


async def _workspace_gate(t: _Env) -> str:
    repo = t.repo(CALC)
    t.script({"implementer": FIX, "debugger": FIX, "reviewer": APPROVE})
    rt = t.runtime(repo)
    try:
        task = await rt.submit("The test_add test is failing; fix add() in calc.py")
        final = await rt.run_task(task.id)
        _expect(final.status == "completed", f"task {final.status}: {final.status_reason}")
        _expect(final.evidence_level == "strong", f"evidence level {final.evidence_level}")
        _expect(
            _git_status(repo) == "M calc.py",
            f"canonical checkout changed unexpectedly: {_git_status(repo)!r}",
        )
        return f"isolated fix verified by real pytest, reviewed, gated ({final.evidence_level}) and applied"
    finally:
        await rt.close()


async def _fail_closed(t: _Env) -> str:
    repo = t.repo(CALC)
    t.script({"implementer": LIAR, "debugger": LIAR, "reviewer": APPROVE})
    rt = t.runtime(repo)
    try:
        task = await rt.submit("The test_add test is failing; fix add() in calc.py")
        final = await rt.run_task(task.id)
        _expect(final.status == "failed", f"a claim without evidence ended {final.status}")
        _expect(_git_status(repo) == "", "canonical checkout was modified")
        return "a success claim without evidence fails closed; checkout untouched"
    finally:
        await rt.close()


async def _wait_status(rt: Any, task_id: str, statuses: set[str], timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = rt.tasks.get(task_id)
        if task.status.value in statuses:
            return task
        await asyncio.sleep(0.1)
    raise AssertionError(f"task stayed {rt.tasks.get(task_id).status.value} (expected {sorted(statuses)})")


async def _cancellation(t: _Env) -> str:
    repo = t.repo(CALC)
    t.script({"implementer": SLOW, "reviewer": APPROVE})
    rt = t.runtime(repo)
    try:
        task = await rt.submit("Fix add() in calc.py so it adds", mode="direct")
        runner = rt.start_task(task.id)
        await _wait_status(rt, task.id, {"running"}, 20)
        started = time.monotonic()
        rt.cancel_task(task.id, reason="selftest cancellation")
        final = await asyncio.wait_for(runner, timeout=30)
        took = time.monotonic() - started
        _expect(final.status == "cancelled", f"task ended {final.status}")
        _expect(_git_status(repo) == "", "cancelled work reached the checkout")
        return f"in-flight model call cancelled durably in {took:.1f}s"
    finally:
        await rt.close()


async def _crash_recovery(t: _Env) -> str:
    repo = t.repo(CALC)
    t.script({"implementer": SLOW, "reviewer": APPROVE})
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CRASH_CHILD,
        str(repo),
        env=t.env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert child.stdout is not None
        task_id = (await asyncio.wait_for(child.stdout.readline(), timeout=60)).decode().strip()
        if not task_id.startswith("tsk_"):
            err = (await child.stderr.read()).decode(errors="replace") if child.stderr else ""
            raise AssertionError(f"child did not start a task: {task_id!r} {err[-500:]}")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            conn = sqlite3.connect(f"file:{t.paths.db_path}?mode=ro", uri=True, timeout=5)
            try:
                busy = conn.execute(
                    "SELECT COUNT(*) FROM model_calls WHERE task_id = ? AND status = 'running'", (task_id,)
                ).fetchone()[0]
            finally:
                conn.close()
            if busy:
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("the child never reached its model call")
        os.killpg(child.pid, signal.SIGKILL)
        await child.wait()
    finally:
        if child.returncode is None:
            os.killpg(child.pid, signal.SIGKILL)
            await child.wait()
    t.script({"implementer": FIX, "reviewer": APPROVE})
    rt = t.runtime(repo)
    try:
        report = rt.last_recovery
        _expect(
            report is not None and any(e["task"] == task_id for e in report.interrupted_tasks),
            f"recovery did not find the interrupted task: {report.to_dict() if report else None}",
        )
        _expect(rt.tasks.get(task_id).status == "interrupted", f"task is {rt.tasks.get(task_id).status}")
        rt.tasks.resume(task_id, note="selftest resume after crash")
        final = await rt.run_task(task_id)
        _expect(final.status == "completed", f"resumed task ended {final.status}: {final.status_reason}")
        _expect(
            final.recovered_count >= 1 or final.attempt_count >= 2,
            "the resumed run was not recorded as a new attempt",
        )
        return f"SIGKILL mid-call → attempt interrupted, lease reclaimed, resumed as attempt {final.attempt_count} and completed"
    finally:
        await rt.close()


async def _evals(t: _Env) -> str:
    from coremain.evals.harness import EvalHarness

    rt = t.runtime()
    try:
        ids = ["question-where-defined", "provider-transient-retry", "security-secret-leak-blocked"]
        run = await EvalHarness(rt).run(suite="smoke", scenario_ids=ids, label="selftest")
        bad = [f"{r['scenario']}: {r['status']}" for r in run["results"] if r["status"] != "pass"]
        _expect(not bad, f"scenarios failed: {bad}")
        return f"{run['summary']['passed']}/{run['summary']['total']} smoke scenarios passed in sandboxes"
    finally:
        await rt.close()


async def _live_call(user_env: Mapping[str, str]) -> str:
    from coremain.providers.client import CallContext
    from coremain.providers.types import ChatMessage, ChatRequest
    from coremain.routing.router import RouteRequirements
    from coremain.runtime.app import CoreRuntime
    from coremain.runtime.cancel import CancelToken

    rt = CoreRuntime.open(
        paths=resolve_core_paths(user_env), env=user_env, mode="selftest", auto_recover=False
    )
    try:
        _expect(bool(rt.registry.models()), "no models configured (add a provider and model first)")
        model = rt.router.route(
            RouteRequirements(role="summarizer", task_kind="question", needs_tools=False)
        ).model
        req = ChatRequest(
            model=model.config.id,
            messages=[ChatMessage("user", "Reply with exactly one word: ready")],
            max_output_tokens=16,
            metadata={"role": "summarizer", "node": "selftest"},
        )
        started = time.monotonic()
        resp = await rt.client.complete(
            model, req, CallContext(role="summarizer", purpose="selftest"), cancel=CancelToken()
        )
        _expect(bool(resp.text.strip()), "the model returned an empty response")
        return f"{model.ref} answered in {time.monotonic() - started:.1f}s ({resp.usage.input_tokens}+{resp.usage.output_tokens} tokens)"
    finally:
        await rt.close()


async def run_selftest(
    *,
    live: bool = False,
    keep: bool = False,
    user_env: Mapping[str, str] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    user_env = dict(os.environ if user_env is None else user_env)
    base = Path(tempfile.mkdtemp(prefix="core-selftest-"))
    t = _Env(base, user_env)
    steps: list[tuple[str, Callable[[], Awaitable[str]]]] = [
        ("storage & migrations", lambda: _storage(t)),
        ("policy invariants", lambda: _policy(t)),
        ("secret handling", lambda: _secrets(t)),
        ("isolated workspace + evidence gate", lambda: _workspace_gate(t)),
        ("fail-closed completion", lambda: _fail_closed(t)),
        ("durable cancellation", lambda: _cancellation(t)),
        ("crash recovery", lambda: _crash_recovery(t)),
        ("evaluation harness", lambda: _evals(t)),
    ]
    if live:
        steps.append(("live model call", lambda: _live_call(user_env)))
    checks = []
    try:
        for name, fn in steps:
            started = time.monotonic()
            try:
                detail, ok = await fn(), True
            except Exception as exc:  # noqa: BLE001 - every failure becomes a reported check
                detail, ok = f"{type(exc).__name__}: {exc}", False
            checks.append(
                {"name": name, "ok": ok, "detail": detail, "duration_s": round(time.monotonic() - started, 2)}
            )
            if progress is not None:
                progress(name, ok, detail)
    finally:
        if not keep:
            shutil.rmtree(base, ignore_errors=True)
    failed = sum(1 for c in checks if not c["ok"])
    return {
        "checks": checks,
        "total": len(checks),
        "passed": len(checks) - failed,
        "failed": failed,
        "dir": str(base),
    }
