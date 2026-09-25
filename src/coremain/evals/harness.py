"""Evaluation harness: run scenarios in disposable sandboxes and record comparable outcomes.

Each scenario gets a fresh git repository and a fresh Core Main home (database, workspaces,
config), so evaluation never touches the user's projects or history; only the run/result
records are written to the user's database. Runs are non-interactive: approval requests are
denied and questions answered with "no user available", both of which are recorded.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w

from coremain.errors import UsageError
from coremain.evals.checks import CheckContext, evaluate
from coremain.evals.scenarios import Scenario, expand, load_scenarios, regression_scenario
from coremain.paths import ProjectPaths, resolve_core_paths
from coremain.tools.base import ApprovalDecision
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, loads
from coremain.version import __version__

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime

_IDENT = ["-c", "user.name=Core Main Eval", "-c", "user.email=eval@localhost", "-c", "commit.gpgsign=false"]
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR")


async def _deny_approval(approval: Any) -> ApprovalDecision:
    return ApprovalDecision(False, reason="evaluation runs are non-interactive; approvals are denied")


async def _no_user(question: str, context: str | None) -> str:
    return "No user is available during this evaluation run. Proceed with your best judgment and state assumptions."


def _git_state(repo: Path) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False
    ).stdout
    return head, status


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = expand(value) if isinstance(value, str) else value
    return base


class EvalHarness:
    def __init__(self, rt: CoreRuntime):
        self.rt = rt

    # ------------------------------------------------------------ selection
    def available(self) -> list[Scenario]:
        dirs = [self.rt.paths.config_dir / "evals"]
        if self.rt.project_root is not None:
            dirs.append(ProjectPaths(self.rt.project_root).core_dir / "evals")
        scenarios = load_scenarios([d for d in dirs if d.is_dir()])
        for row in self.rt.db.query("SELECT * FROM regression_cases WHERE status = 'active'"):
            case = dict(row)
            case["scenario"] = loads(case.pop("scenario_json"), {})
            with contextlib.suppress(Exception):
                scenarios.append(regression_scenario(case))
        return scenarios

    def select(self, suite: str, ids: list[str] | None) -> list[Scenario]:
        scenarios = self.available()
        if ids:
            by_id = {s.id: s for s in scenarios}
            unknown = [i for i in ids if i not in by_id]
            if unknown:
                raise UsageError(
                    f"unknown scenario(s): {', '.join(unknown)}", hint="list them with `core eval list`"
                )
            return [by_id[i] for i in ids]
        chosen = [s for s in scenarios if suite in s.suites]
        if not chosen:
            raise UsageError(
                f"suite '{suite}' has no scenarios", hint="suites: smoke (scripted), core (live), regression"
            )
        return chosen

    def active_heuristics(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.rt.db.query("SELECT * FROM heuristics WHERE status = 'active'")]

    # ------------------------------------------------------------------ run
    async def run(
        self,
        *,
        suite: str = "smoke",
        scenario_ids: list[str] | None = None,
        model: str | None = None,
        mode: str | None = None,
        label: str | None = None,
        keep: bool = False,
        on_result: Callable[[dict[str, Any]], Any] | None = None,
        heuristics: list[dict[str, Any]] | None = None,
        record: bool = True,
    ) -> dict[str, Any]:
        scenarios = self.select(suite, scenario_ids)
        scripted = model is None and suite == "smoke"
        if not scripted:
            if model is not None:
                self.rt.registry.model(model)
            elif not self.rt.registry.models():
                raise UsageError(
                    "no models are configured for a live evaluation",
                    hint="configure a provider, or use --suite smoke",
                )
        heuristics = self.active_heuristics() if heuristics is None else heuristics
        strategy = {
            "scripted": scripted,
            "model": model,
            "mode": mode,
            "routing": None
            if scripted or model
            else self.rt.config.routing.model_dump(mode="json", exclude_none=True),
            "heuristics": sorted(f"{h['name']}@v{h['version']}" for h in heuristics),
        }
        from coremain.agent.prompts import PROMPT_VERSION

        run_id = new_id("evr", now=self.rt.clock.now())
        started = self.rt.clock.now()
        if record:
            self.rt.db.execute(
                "INSERT INTO eval_runs(id, suite, label, strategy_json, core_version, prompt_version, heuristics_json, status, "
                "summary_json, started_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    suite,
                    label,
                    dumps(strategy),
                    __version__,
                    PROMPT_VERSION,
                    dumps(strategy["heuristics"]),
                    "running",
                    "{}",
                    started,
                ),
            )
        results: list[dict[str, Any]] = []
        try:
            for sc in scenarios:
                result = await self.run_scenario(
                    sc, scripted=scripted, model=model, mode=mode, keep=keep, heuristics=heuristics
                )
                result["run_id"] = run_id
                results.append(result)
                if record:
                    self.rt.db.execute(
                        "INSERT INTO eval_results(id, run_id, scenario, family, status, task_status, metrics_json, checks_json, error, "
                        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            new_id("evx", now=self.rt.clock.now()),
                            run_id,
                            sc.id,
                            sc.family,
                            result["status"],
                            result["task_status"],
                            dumps(result["metrics"]),
                            dumps(result["checks"]),
                            result.get("error"),
                            self.rt.clock.now(),
                        ),
                    )
                if on_result is not None:
                    on_result(result)
        finally:
            summary = summarize(results)
            if record:
                self.rt.db.execute(
                    "UPDATE eval_runs SET status = ?, summary_json = ?, ended_at = ? WHERE id = ?",
                    (
                        "completed" if len(results) == len(scenarios) else "aborted",
                        dumps(summary),
                        self.rt.clock.now(),
                        run_id,
                    ),
                )
        label_text = "scripted" if scripted else (f"model={model}" if model else "configured routing")
        return {
            "id": run_id,
            "suite": suite,
            "strategy": label_text,
            "strategy_detail": strategy,
            "summary": summary,
            "results": results,
        }

    # ------------------------------------------------------------- scenario
    async def run_scenario(
        self,
        sc: Scenario,
        *,
        scripted: bool,
        model: str | None,
        mode: str | None,
        keep: bool,
        heuristics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        base = {
            "scenario": sc.id,
            "family": sc.family,
            "title": sc.title,
            "task_status": None,
            "checks": [],
            "metrics": {},
        }
        missing = sc.missing_requirements()
        if missing:
            return {**base, "status": "skipped", "error": f"missing requirement(s): {', '.join(missing)}"}
        if scripted and not sc.script:
            return {**base, "status": "skipped", "error": "scenario has no script (live-only)"}
        work = Path(tempfile.mkdtemp(prefix=f"core-eval-{sc.id}-"))
        t0 = time.monotonic()
        try:
            repo = await asyncio.to_thread(self._make_repo, work / "repo", sc)
            env = self._env(work)
            paths = resolve_core_paths(env)
            paths.ensure()
            self._write_config(
                paths.config_file, sc, scripted=scripted, work=work, credentials=paths.credentials_file
            )
            from coremain.runtime.app import CoreRuntime

            sandbox = CoreRuntime.open(project_root=repo, paths=paths, env=env, mode="eval")
            try:
                self._seed_heuristics(sandbox, heuristics)
                sandbox.approval_handler = _deny_approval
                sandbox.input_handler = _no_user
                await sandbox.start()
                initial_head, initial_status = await asyncio.to_thread(_git_state, repo)
                task = await sandbox.submit(
                    sc.request,
                    mode=mode or sc.mode,
                    model=None if scripted else model,
                    contract=sc.contract or None,
                )
                runner = sandbox.start_task(task.id)
                done, _ = await asyncio.wait({runner}, timeout=sc.timeout_s)
                if not done:
                    sandbox.cancel_task(task.id, reason=f"evaluation timeout after {sc.timeout_s:.0f}s")
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(runner, timeout=60)
                final = sandbox.tasks.get(task.id)
                ctx = CheckContext(sandbox, final, repo, initial_head, initial_status, env)
                checks = [await evaluate(check, ctx) for check in sc.checks]
                metrics = self._metrics(sandbox, final, time.monotonic() - t0)
            finally:
                await sandbox.close()
            status = "pass" if all(c["ok"] for c in checks) else "fail"
            error = None if done else "timeout"
            return {
                **base,
                "status": status,
                "task_status": final.status.value,
                "checks": checks,
                "metrics": metrics,
                "error": error,
                "workdir": str(work) if keep else None,
            }
        except Exception as exc:  # noqa: BLE001 - one broken scenario must not abort the run
            return {
                **base,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": {"duration_s": round(time.monotonic() - t0, 2)},
                "workdir": str(work) if keep else None,
            }
        finally:
            if not keep:
                shutil.rmtree(work, ignore_errors=True)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _make_repo(repo: Path, sc: Scenario) -> Path:
        repo.mkdir(parents=True)
        for rel, content in sc.files.items():
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            if rel.endswith(".sh"):
                path.chmod(0o755)
        run = lambda *a: subprocess.run(["git", *_IDENT, *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
        run("init", "-q", "-b", "main")
        run("add", "-A")
        run("commit", "-qm", "scenario fixture", "--allow-empty")
        for rel, content in sc.uncommitted.items():
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return repo

    def _env(self, work: Path) -> dict[str, str]:
        home = work / "home"
        home.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in self.rt.env.items() if k in _ENV_KEEP}
        for key in {r.removeprefix("env:") for r in self._env_refs()}:
            if key in self.rt.env:
                env[key] = self.rt.env[key]
        env.update({"HOME": str(home), "CORE_HOME": str(work / "core")})
        return env

    def _env_refs(self) -> list[str]:
        refs = [p.api_key for p in self.rt.config.providers.values() if p.api_key]
        refs += [v for p in self.rt.config.providers.values() for v in p.secret_headers.values()]
        return [r for r in refs if r.startswith("env:")]

    def _write_config(
        self, path: Path, sc: Scenario, *, scripted: bool, work: Path, credentials: Path
    ) -> None:
        cfg: dict[str, Any] = {}
        if scripted:
            script = work / "script.json"
            script.write_text(json.dumps(sc.scripted_turns()), encoding="utf-8")
            cfg["providers"] = {"eval": {"kind": "scripted", "script": str(script)}}
            cfg["models"] = {}
            for i in range(max(1, sc.models)):
                key = "scripted" if i == 0 else f"scripted-{chr(ord('a') + i)}"
                cfg["models"][key] = {
                    "provider": "eval",
                    "id": key,
                    "context_window": 128_000,
                    "max_output_tokens": 8192,
                }
        else:
            live = self.rt.config.model_dump(
                mode="json", include={"providers", "models", "routing", "budgets"}, exclude_none=True
            )
            cfg.update(live)
            user_credentials = self.rt.paths.credentials_file
            if user_credentials.exists():
                credentials.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(user_credentials, credentials)
        test_command = sc.expanded_test_command()
        if test_command:
            cfg.setdefault("verification", {}).setdefault("commands", {})["test"] = test_command
        _deep_merge(cfg, sc.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(cfg), encoding="utf-8")

    @staticmethod
    def _seed_heuristics(sandbox: CoreRuntime, heuristics: list[dict[str, Any]]) -> None:
        for h in heuristics:
            sandbox.db.execute(
                "INSERT OR REPLACE INTO heuristics(id, name, version, kind, content_json, rationale, source_failure_ids_json, status, "
                "validation_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    h["id"],
                    h["name"],
                    h["version"],
                    h["kind"],
                    h["content_json"],
                    h["rationale"],
                    h.get("source_failure_ids_json") or "[]",
                    "active",
                    h.get("validation_json") or "{}",
                    h["created_at"],
                    h["updated_at"],
                ),
            )
        if heuristics:
            sandbox.router.heuristics = sandbox._active_heuristics("routing")

    @staticmethod
    def _metrics(sandbox: CoreRuntime, task: Any, duration: float) -> dict[str, Any]:
        calls = sandbox.db.one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(input_tokens),0) AS tin, COALESCE(SUM(output_tokens),0) AS tout, "
            "SUM(cost_usd) AS cost, COALESCE(SUM(retries),0) AS retries, SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors "
            "FROM model_calls WHERE task_id IN (SELECT id FROM tasks)"
        )
        tools = sandbox.db.one(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS failed FROM tool_calls"
        )
        models = [
            r["model_key"]
            for r in sandbox.db.query(
                "SELECT DISTINCT model_key FROM model_calls WHERE model_key IS NOT NULL"
            )
        ]
        return {
            "duration_s": round(duration, 2),
            "model_calls": int(calls["n"]) if calls else 0,
            "model_errors": int(calls["errors"] or 0) if calls else 0,
            "input_tokens": int(calls["tin"]) if calls else 0,
            "output_tokens": int(calls["tout"]) if calls else 0,
            "cost_usd": round(float(calls["cost"]), 6) if calls and calls["cost"] is not None else None,
            "retries": int(calls["retries"]) if calls else 0,
            "tool_calls": int(tools["n"]) if tools else 0,
            "tool_failures": int(tools["failed"] or 0) if tools else 0,
            "attempts": task.attempt_count,
            "evidence_level": task.evidence_level,
            "workflow": (task.decision or {}).get("workflow"),
            "models": models,
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    counted = [r for r in results if r["status"] != "skipped"]
    passed = sum(1 for r in counted if r["status"] == "pass")
    families: dict[str, dict[str, int]] = {}
    for r in counted:
        fam = families.setdefault(r["family"], {"total": 0, "passed": 0})
        fam["total"] += 1
        fam["passed"] += r["status"] == "pass"
    durations = [r["metrics"].get("duration_s") or 0 for r in counted]
    return {
        "total": len(counted),
        "passed": passed,
        "failed": sum(1 for r in counted if r["status"] == "fail"),
        "errors": sum(1 for r in counted if r["status"] == "error"),
        "skipped": len(results) - len(counted),
        "pass_rate": passed / len(counted) if counted else 0.0,
        "families": families,
        "duration_s": round(sum(durations), 2),
        "model_calls": sum(r["metrics"].get("model_calls") or 0 for r in counted),
        "tokens": sum(
            (r["metrics"].get("input_tokens") or 0) + (r["metrics"].get("output_tokens") or 0)
            for r in counted
        ),
    }
