"""Runs planned verification checks in a task workspace and records bound evidence."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from coremain.config.schema import CoreConfig
from coremain.exec.process import ProcessRunner
from coremain.runtime.cancel import CancelToken
from coremain.security.commands import analyze_command
from coremain.security.env import build_subprocess_env
from coremain.security.paths import sensitive_reason
from coremain.security.policy import Capability, PolicyEngine, PolicyRequest
from coremain.security.redact import Redactor
from coremain.store.artifacts import ArtifactStore
from coremain.verify.evidence import EvidenceEngine, EvidenceKind, Requirement, Trust
from coremain.verify.testparse import parse_test_output
from coremain.workspaces.manager import Workspace, WorkspaceDiff, WorkspaceManager

PROTECTED_PATTERNS = (".core/config.toml", ".core/config.local.toml", ".core/policy*", ".git/*")


@dataclass
class CheckResult:
    kind: str
    name: str
    status: str
    summary: str
    evidence_id: str | None = None
    output_excerpt: str = ""
    not_applicable: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "pass" or (self.status == "skipped" and self.not_applicable)


@dataclass
class VerificationReport:
    fingerprint: str
    diff_hash: str
    results: list[CheckResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def required_failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok and r.name != "optional"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "diff_hash": self.diff_hash,
            "notes": self.notes,
            "results": [r.__dict__ for r in self.results],
        }


class VerificationRunner:
    def __init__(
        self,
        *,
        config: CoreConfig,
        processes: ProcessRunner,
        evidence: EvidenceEngine,
        artifacts: ArtifactStore,
        workspaces: WorkspaceManager,
        policy: PolicyEngine,
        redactor: Redactor,
        base_env: dict[str, str] | None = None,
        browser: Callable[[], Any] | None = None,
    ):
        self.config = config
        self.processes = processes
        self.evidence = evidence
        self.artifacts = artifacts
        self.workspaces = workspaces
        self.policy = policy
        self.redactor = redactor
        self.base_env = base_env
        # Lazily resolves the browser extension (optional dependency) only when a check needs it.
        self.browser = browser

    async def run(
        self,
        *,
        project_id: str,
        task_id: str,
        attempt_id: str,
        workspace: Workspace,
        requirements: list[Requirement],
        profile: dict[str, Any],
        contract: dict[str, Any],
        cancel: CancelToken,
        fence: Any = None,
        extra_env: dict[str, str] | None = None,
    ) -> VerificationReport:
        diff = await self.workspaces.diff(workspace)
        report = VerificationReport(diff.fingerprint, diff.diff_hash)
        common = {
            "project_id": project_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "workspace_id": workspace.id,
            "fingerprint": diff.fingerprint,
            "diff_hash": diff.diff_hash,
            "fence": fence,
        }
        commands = {**(profile.get("commands") or {}), **self.config.verification.commands}
        kinds = {r.kind for r in requirements}
        if EvidenceKind.DIFF in kinds:
            status = "pass" if not diff.empty else "fail"
            listed = ", ".join(f"{f.status} {f.path}" for f in diff.files[:8])
            more = f", … {len(diff.files) - 8} more" if len(diff.files) > 8 else ""
            summary = (
                f"{diff.stats.get('files', 0)} file(s) changed (+{diff.stats.get('added', 0)} -{diff.stats.get('removed', 0)}): "
                f"{listed}{more}"
                if not diff.empty
                else "no changes in workspace"
            )
            ev = self.evidence.record(
                kind=EvidenceKind.DIFF,
                status=status,
                trust=Trust.VERIFIED,
                summary=summary,
                data={"files": [f"{f.status} {f.path}" for f in diff.files][:200]},
                **common,
            )
            report.results.append(CheckResult(EvidenceKind.DIFF, "", status, summary, ev.id))
        if EvidenceKind.SCOPE in kinds:
            report.results.append(self._scope(diff, contract, common))
        if EvidenceKind.NO_SECRETS in kinds:
            report.results.append(self._secrets(diff, common))
        if EvidenceKind.SYNTAX in kinds:
            report.results.append(await self._syntax(workspace, diff, common))
        env = build_subprocess_env(
            self.base_env if self.base_env is not None else os.environ,
            passthrough=self.config.permissions.env_passthrough,
            extra={"CI": "1", **(extra_env or {})},
        )
        for req in requirements:
            cancel.raise_if_cancelled()
            if req.kind == EvidenceKind.TESTS:
                if req.name == "suite" and commands.get("test"):
                    report.results.append(
                        await self._command(req, commands["test"], workspace, env, common, cancel, tests=True)
                    )
                elif req.name.startswith("required:") and commands.get("test"):
                    target = req.name.split(":", 1)[1]
                    report.results.append(
                        await self._command(
                            req,
                            f"{commands['test']} {shlex.quote(target)}",
                            workspace,
                            env,
                            common,
                            cancel,
                            tests=True,
                        )
                    )
            elif req.kind in (EvidenceKind.LINT, EvidenceKind.TYPECHECK, EvidenceKind.BUILD):
                cmd = commands.get(str(req.kind))
                if cmd:
                    if req.kind == EvidenceKind.LINT:
                        cmd = self._narrow_lint(cmd, diff)
                    report.results.append(await self._command(req, cmd, workspace, env, common, cancel))
            elif req.kind == EvidenceKind.COMMAND:
                cmd = next(
                    (
                        c.get("command") if isinstance(c, dict) else str(c)
                        for c in contract.get("required_commands", [])
                        if (c.get("name") if isinstance(c, dict) else None) == req.name
                        or (c.get("command") if isinstance(c, dict) else str(c)) == req.name
                    ),
                    req.name,
                )
                report.results.append(await self._command(req, str(cmd), workspace, env, common, cancel))
        browser_reqs = [r for r in requirements if r.kind == EvidenceKind.BROWSER]
        if browser_reqs:
            report.results.extend(await self._browser(browser_reqs, contract, workspace, common, cancel))
        after = await self.workspaces.fingerprint(workspace)
        if after != diff.fingerprint:
            changed = await self._changed_between(workspace, diff.fingerprint, after)
            note = (
                "verification commands modified the workspace ("
                + ", ".join(changed[:8])
                + "); evidence is bound to the pre-verification state and will be reported stale"
            )
            report.notes.append(note)
        return report

    async def _changed_between(self, ws: Workspace, a: str, b: str) -> list[str]:
        res = await self.workspaces.git_for(ws).run("diff", "--name-only", a, b, check=False)
        return [line for line in res.text.splitlines() if line]

    async def _browser(
        self,
        reqs: list[Requirement],
        contract: dict[str, Any],
        workspace: Workspace,
        common: dict[str, Any],
        cancel: CancelToken,
    ) -> list[CheckResult]:
        checks = [
            c if isinstance(c, dict) else {"name": str(c), "url": str(c)}
            for c in contract.get("browser_checks") or []
        ]
        by_name = {str(c.get("name", "browser")): c for c in checks}
        wanted = [by_name[r.name] for r in reqs if r.name in by_name]
        results: list[dict[str, Any]] = []
        error: str | None = None
        try:
            if self.browser is None:
                raise RuntimeError("browser automation is not available in this runtime")
            manager = self.browser()
            results = await manager.run_checks(
                wanted,
                workspace_path=workspace.path,
                task_id=common["task_id"],
                attempt_id=common["attempt_id"],
                cancel=cancel,
            )
        except Exception as exc:  # noqa: BLE001 - an unavailable browser is recorded as an error, never as a pass
            error = f"{type(exc).__name__}: {exc}"
        out = []
        for i, check in enumerate(wanted):
            name = str(check.get("name", "browser"))
            res = (
                results[i]
                if i < len(results)
                else {"status": "error", "summary": error or "check did not run"}
            )
            status = res.get("status", "error")
            ev = self.evidence.record(
                kind=EvidenceKind.BROWSER,
                status=status if status in {"pass", "fail", "error"} else "error",
                trust=Trust.OBSERVED,
                summary=f"{name}: {res.get('summary', '')}"[:500],
                artifact_id=res.get("artifact_id"),
                data={"name": name, "check": check, "console_errors": res.get("console_errors", [])[:20]},
                **common,
            )
            out.append(CheckResult(EvidenceKind.BROWSER, name, ev.status, ev.summary, ev.id))
        return out

    def _scope(self, diff: WorkspaceDiff, contract: dict[str, Any], common: dict[str, Any]) -> CheckResult:
        forbidden = list(contract.get("forbidden_paths") or [])
        violations = []
        for f in diff.files:
            if any(fnmatch.fnmatch(f.path, pat) for pat in forbidden):
                violations.append(f"{f.path} (forbidden by contract)")
            elif any(fnmatch.fnmatch(f.path, pat) for pat in PROTECTED_PATTERNS):
                violations.append(f"{f.path} (protected Core Main/VCS file)")
            elif f.status != "D" and sensitive_reason(f.path):
                violations.append(f"{f.path} (sensitive file)")
        status = "fail" if violations else "pass"
        summary = (
            "scope violations: " + ", ".join(violations[:6])
            if violations
            else "no forbidden or protected paths modified"
        )
        ev = self.evidence.record(
            kind=EvidenceKind.SCOPE,
            status=status,
            trust=Trust.VERIFIED,
            summary=summary,
            data={"violations": violations},
            **common,
        )
        return CheckResult(EvidenceKind.SCOPE, "", status, summary, ev.id)

    def _secrets(self, diff: WorkspaceDiff, common: dict[str, Any]) -> CheckResult:
        added = "\n".join(
            line[1:]
            for line in diff.patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        findings = self.redactor.scan(added)
        kinds = sorted(
            {f.kind for f in findings if f.kind != "known_secret"}
            | ({"known_secret"} if any(f.kind == "known_secret" for f in findings) else set())
        )
        status = "fail" if findings else "pass"
        summary = (
            f"possible secrets added: {', '.join(kinds)}"
            if findings
            else "no secrets detected in added lines"
        )
        ev = self.evidence.record(
            kind=EvidenceKind.NO_SECRETS,
            status=status,
            trust=Trust.VERIFIED,
            summary=summary,
            data={"count": len(findings), "kinds": kinds},
            **common,
        )
        return CheckResult(EvidenceKind.NO_SECRETS, "", status, summary, ev.id)

    async def _syntax(self, ws: Workspace, diff: WorkspaceDiff, common: dict[str, Any]) -> CheckResult:
        problems: list[str] = []
        checked = 0
        for f in diff.files:
            if f.status == "D":
                continue
            path = ws.path / f.path
            if f.path.endswith(".py"):
                checked += 1
                try:
                    compile(path.read_text(encoding="utf-8", errors="replace"), f.path, "exec")
                except SyntaxError as exc:
                    problems.append(f"{f.path}:{exc.lineno}: {exc.msg}")
            elif f.path.endswith((".js", ".mjs", ".cjs")) and await asyncio.to_thread(
                lambda: bool(__import__("shutil").which("node"))
            ):
                checked += 1
                proc = await asyncio.create_subprocess_exec(
                    "node",
                    "--check",
                    str(path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, err = await proc.communicate()
                if proc.returncode != 0:
                    problems.append(
                        f"{f.path}: {err.decode(errors='replace').strip().splitlines()[-1] if err else 'syntax error'}"
                    )
            elif f.path.endswith(".json"):
                checked += 1
                import json

                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    problems.append(f"{f.path}: {exc}")
        # With no changed file in a checkable language the requirement is vacuous: recorded as a
        # not-applicable skip that satisfies the gate but adds no evidence strength.
        status = "fail" if problems else ("pass" if checked else "skipped")
        summary = (
            "; ".join(problems[:5])
            if problems
            else (
                f"{checked} changed file(s) parse"
                if checked
                else "no changed file has a syntax checker (not applicable)"
            )
        )
        ev = self.evidence.record(
            kind=EvidenceKind.SYNTAX,
            status=status,
            trust=Trust.OBSERVED,
            summary=summary,
            data={"checked": checked, "problems": problems, "not_applicable": not checked and not problems},
            **common,
        )
        return CheckResult(
            EvidenceKind.SYNTAX, "", status, summary, ev.id, not_applicable=not checked and not problems
        )

    @staticmethod
    def _narrow_lint(cmd: str, diff: WorkspaceDiff) -> str:
        py = [f.path for f in diff.files if f.status != "D" and f.path.endswith(".py")]
        if py and re.search(r"\bruff check \.$", cmd):
            return cmd[:-1] + " ".join(shlex.quote(p) for p in py[:200])
        return cmd

    async def _command(
        self,
        req: Requirement,
        cmd: str,
        ws: Workspace,
        env: dict[str, str],
        common: dict[str, Any],
        cancel: CancelToken,
        *,
        tests: bool = False,
    ) -> CheckResult:
        analysis = analyze_command(cmd, workspace=ws.path)
        decision = self.policy.evaluate(
            PolicyRequest(
                capability=Capability.EXEC,
                target=cmd,
                workspace_root=ws.path,
                workspace_isolated=ws.isolated,
                analysis=analysis,
                project_id=common["project_id"],
                task_id=common["task_id"],
                tool="verification",
            )
        )
        data: dict[str, Any] = {"name": req.name, "command": cmd, "risk": analysis.max_risk.value}
        if decision.decision != "allow":
            summary = f"verification command not run: {decision.decision} by policy ({decision.reason})"
            ev = self.evidence.record(
                kind=req.kind,
                status="error",
                trust=Trust.OBSERVED,
                summary=summary,
                command=cmd,
                data=data,
                tool="verification",
                **common,
            )
            return CheckResult(req.kind, req.name, "error", summary, ev.id)
        result = await self.processes.run(
            shell_command=cmd,
            cwd=ws.path,
            env=env,
            timeout_s=float(self.config.verification.timeout_s),
            cancel=cancel.child(),
            task_id=common["task_id"],
            attempt_id=common["attempt_id"],
        )
        output = result.stdout + ("\n" + result.stderr if result.stderr else "")
        art = self.artifacts.put_text(
            output,
            kind="verification_output",
            name=f"{req.kind}:{req.name or cmd[:40]}",
            project_id=common["project_id"],
            task_id=common["task_id"],
            attempt_id=common["attempt_id"],
        )
        parsed = parse_test_output(output) if tests else None
        if result.status == "timeout":
            status, summary = "error", f"timed out after {self.config.verification.timeout_s}s: {cmd}"
        elif result.exit_code == 0:
            status = "pass"
            summary = parsed.line() if parsed else f"`{cmd}` succeeded"
            if tests and parsed is not None and not parsed.ran_any:
                status, summary = "inconclusive", f"`{cmd}` ran no tests"
        elif tests and result.exit_code == 5 and "pytest" in cmd:
            status, summary = "inconclusive", "pytest collected no tests (exit code 5)"
        else:
            status = "fail"
            summary = parsed.line() if parsed else f"`{cmd}` exited with {result.exit_code}"
        if parsed is not None:
            data["tests"] = parsed.__dict__
        data["exit_code"] = result.exit_code
        data["duration_s"] = round(result.duration_s, 2)
        ev = self.evidence.record(
            kind=req.kind,
            status=status,
            trust=Trust.OBSERVED,
            summary=summary,
            command=cmd,
            exit_code=result.exit_code,
            artifact_id=art.id,
            tool="verification",
            data=data,
            **common,
        )
        tail = "\n".join(output.strip().splitlines()[-40:])
        return CheckResult(req.kind, req.name, status, summary, ev.id, self.redactor.redact(tail))
