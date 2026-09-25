"""CoreRuntime: the single authoritative application runtime (composition root).

CLI, TUI and the stdio JSON-RPC server are clients of this object; none of them hold domain
state of their own. All durable state lives in one SQLite database; live updates flow through
the event bus (in-process) and the events table (cross-process).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from coremain.config.loader import EffectiveConfig, load_config, project_config_hash
from coremain.context.compiler import CompiledContext, ContextCompiler
from coremain.domain.models import Project, Task
from coremain.domain.states import ATTENTION, TaskStatus
from coremain.errors import CoreError, NotFoundError, UsageError
from coremain.events import EventBus, EventLog
from coremain.exec.process import ProcessRunner
from coremain.intel.index import CodeIndex
from coremain.intel.profile import detect_profile, manifests_hash
from coremain.learning.failures import FailureRecorder
from coremain.memory.decisions import DecisionStore
from coremain.memory.store import MemoryStore
from coremain.paths import CorePaths, ProjectPaths, resolve_core_paths
from coremain.providers.client import ModelClient
from coremain.providers.registry import ProviderRegistry
from coremain.review.engine import ReviewStore
from coremain.routing.modes import classify
from coremain.routing.router import Router
from coremain.routing.stats import ModelStats
from coremain.runtime.approvals import ApprovalService
from coremain.runtime.cancel import CancelToken
from coremain.runtime.leases import LeaseManager
from coremain.runtime.projects import ProjectService, SessionService
from coremain.runtime.recovery import RecoveryReport, recover
from coremain.runtime.tasks import TaskService
from coremain.security.credentials import CredentialStore
from coremain.security.policy import PolicyEngine
from coremain.security.redact import Redactor
from coremain.skills.registry import SkillRegistry, builtin_skills_dir
from coremain.store.artifacts import ArtifactStore
from coremain.store.db import Database
from coremain.store.migrate import migrate
from coremain.tools.base import ApprovalHandler, InputHandler, Tool, ToolServices
from coremain.util.clock import SYSTEM_CLOCK, Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps
from coremain.util.text import one_line
from coremain.verify.evidence import EvidenceEngine, EvidenceKind
from coremain.verify.planner import plan_requirements, requirements_to_dicts
from coremain.verify.runner import VerificationRunner
from coremain.version import __version__
from coremain.workspaces.manager import Workspace, WorkspaceManager

log = logging.getLogger(__name__)
RUNTIME_HEARTBEAT_S = 15.0


class CoreRuntime:
    def __init__(
        self,
        *,
        paths: CorePaths,
        db: Database,
        project_root: Path | None,
        overrides: Mapping[str, Any] | None,
        env: Mapping[str, str] | None,
        clock: Clock,
        mode: str,
        transports: dict[str, httpx.AsyncBaseTransport] | None,
    ):
        self.paths = paths
        self.db = db
        self.clock = clock
        self.mode = mode
        self.env = dict(os.environ if env is None else env)
        self.redactor = Redactor()
        self.bus = EventBus()
        self.events = EventLog(db, self.bus, self.redactor, clock)
        self.projects = ProjectService(db, self.events, clock)
        self.sessions = SessionService(db, self.events, clock)
        self.project: Project | None = (
            self.projects.ensure(project_root) if project_root is not None else None
        )
        self.project_root = Path(self.project.root_path) if self.project else None
        cfg_hash = project_config_hash(self.project_root) if self.project_root else None
        self.project_trusted = bool(self.project and ProjectService.is_trusted(self.project, cfg_hash))
        self.effective: EffectiveConfig = load_config(
            paths, self.project_root, overrides=overrides, env=self.env, project_trusted=self.project_trusted
        )
        self.config = self.effective.config
        self.runtime_id = new_id("rt", now=clock.now())
        self.credentials = CredentialStore(paths.credentials_file)
        self.registry = ProviderRegistry(
            self.config, self.credentials, self.redactor, env=self.env, transports=transports
        )
        self.stats = ModelStats(db)
        self.router = Router(
            self.registry,
            self.config.routing,
            self.stats,
            heuristics=self._active_heuristics("routing"),
            clock_now=clock.now,
        )
        self.client = ModelClient(self.registry, db, self.events, clock, self.config)
        # Task workspaces live under the data dir, so protect the specific state locations only.
        protected = [
            paths.config_dir,
            paths.db_path,
            Path(f"{paths.db_path}-wal"),
            Path(f"{paths.db_path}-shm"),
            paths.artifacts_dir,
            paths.snapshots_dir,
            paths.backups_dir,
            paths.data_dir / "indexes",
            paths.imported_skills_dir,
            paths.logs_dir,
        ]
        if self.project_root is not None:
            pp = ProjectPaths(self.project_root)
            protected += [pp.config_file, pp.local_config_file, pp.core_dir / "policy.toml"]
        self.policy = PolicyEngine(self.config.permissions, db=db, clock=clock, protected_paths=protected)
        self.approvals = ApprovalService(db, self.events, self.policy, clock)
        self.leases = LeaseManager(db, clock)
        self.tasks = TaskService(db, self.events, self.leases, clock)
        self.artifacts = ArtifactStore(db, paths.artifacts_dir, self.redactor, clock)
        self.workspaces = WorkspaceManager(
            db, self.events, self.artifacts, paths, clock, self.config.workspace
        )
        self.evidence = EvidenceEngine(db, self.events, self.leases, clock)
        self.reviews = ReviewStore(db, self.events, clock)
        self.memory = MemoryStore(db, self.events, clock, self.config.memory)
        self.decisions = DecisionStore(db, self.events, clock)
        roots: list[tuple[str, Path]] = [
            ("builtin", builtin_skills_dir()),
            ("imported", paths.imported_skills_dir),
            ("user", paths.user_skills_dir),
            ("user", Path(self.env.get("HOME", "~")).expanduser() / ".agents" / "skills"),
        ]
        if self.project_root is not None:
            roots += [
                ("project", ProjectPaths(self.project_root).skills_dir),
                ("project", self.project_root / ".agents" / "skills"),
            ]
        self.skills = SkillRegistry(db, self.config.skills, roots, clock, self.events)
        self.index = CodeIndex(db, self.config.intel, clock)
        self.compiler = ContextCompiler(self.index, self.config.context, self.redactor)
        self.processes = ProcessRunner(
            db,
            clock,
            self.runtime_id,
            max_output_bytes=self.config.exec.max_output_bytes,
            kill_grace_s=self.config.exec.kill_grace_s,
            max_memory_mb=self.config.exec.max_memory_mb,
            spool_dir=paths.tmp_dir,
        )
        self.verifier = VerificationRunner(
            config=self.config,
            processes=self.processes,
            evidence=self.evidence,
            artifacts=self.artifacts,
            workspaces=self.workspaces,
            policy=self.policy,
            redactor=self.redactor,
            base_env=self.env,
            browser=lambda: self.extension("browser"),
        )
        self.failures = FailureRecorder(db, self.events, clock)
        self.approval_handler: ApprovalHandler | None = None
        self.input_handler: InputHandler | None = None
        self._cancels: dict[str, CancelToken] = {}
        self._running: dict[str, asyncio.Task[Task]] = {}
        self._sem: asyncio.Semaphore | None = None
        self._worker_seq = 0
        self._heartbeat: asyncio.Task[None] | None = None
        self._extensions: dict[str, Any] = {}
        self.last_recovery: RecoveryReport | None = None
        for provider_id, cfg in self.config.providers.items():
            if cfg.api_key:
                self.registry.credential_status(provider_id)

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def open(
        cls,
        *,
        project_root: Path | None = None,
        paths: CorePaths | None = None,
        overrides: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
        clock: Clock | None = None,
        mode: str = "cli",
        transports: dict[str, httpx.AsyncBaseTransport] | None = None,
        auto_recover: bool = True,
    ) -> CoreRuntime:
        paths = paths or resolve_core_paths(env)
        paths.ensure()
        db = Database(paths.db_path)
        try:
            migrate(db, backup_dir=paths.backups_dir)
            rt = cls(
                paths=paths,
                db=db,
                project_root=project_root,
                overrides=overrides,
                env=env,
                clock=clock or SYSTEM_CLOCK,
                mode=mode,
                transports=transports,
            )
        except BaseException:
            db.close()
            raise
        now = rt.clock.now()
        db.execute(
            "INSERT INTO runtimes(id, pid, host, version, mode, started_at, heartbeat_at, status) VALUES (?,?,?,?,?,?,?,?)",
            (rt.runtime_id, os.getpid(), socket.gethostname(), __version__, mode, now, now, "running"),
        )
        if auto_recover:
            try:
                rt.last_recovery = recover(rt)
            except CoreError as exc:
                log.warning("automatic recovery failed: %s", exc)
        return rt

    async def start(self) -> None:
        if self._heartbeat is None:
            self._heartbeat = asyncio.create_task(self._runtime_heartbeat())

    async def _runtime_heartbeat(self) -> None:
        while True:
            await asyncio.sleep(RUNTIME_HEARTBEAT_S)
            with contextlib.suppress(Exception):
                self.db.execute(
                    "UPDATE runtimes SET heartbeat_at = ? WHERE id = ?", (self.clock.now(), self.runtime_id)
                )

    async def close(self) -> None:
        for token in list(self._cancels.values()):
            token.cancel("runtime shutting down")
        for task in list(self._running.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=10)
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat
        await self.processes.terminate_all()
        for ext in self._extensions.values():
            closer = getattr(ext, "aclose", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer()
        await self.registry.aclose()
        with contextlib.suppress(Exception):
            self.db.execute(
                "UPDATE runtimes SET status = 'stopped', stopped_at = ? WHERE id = ?",
                (self.clock.now(), self.runtime_id),
            )
        self.db.close()

    def require_project(self) -> Project:
        if self.project is None:
            raise UsageError(
                "this command must be run inside a project directory",
                hint="cd into a repository or pass --project PATH",
            )
        return self.project

    # --------------------------------------------------------------- extensions
    def extension(self, name: str) -> Any:
        """Lazily constructed optional subsystems (mcp, browser, research, lsp)."""
        if name in self._extensions:
            return self._extensions[name]
        ext: Any = None
        if name == "mcp":
            from coremain.mcp.manager import MCPManager

            ext = MCPManager(self)
        elif name == "browser":
            from coremain.browser.manager import BrowserManager

            ext = BrowserManager(self)
        elif name == "research":
            from coremain.research.service import ResearchService

            ext = ResearchService(self)
        elif name == "lsp":
            from coremain.intel.lsp import LSPManager

            ext = LSPManager(self)
        else:
            raise NotFoundError(f"unknown extension {name}")
        self._extensions[name] = ext
        return ext

    def extra_tools(self, role: str) -> list[Tool]:
        tools: list[Tool] = []
        for name in ("research", "browser", "lsp", "mcp"):
            try:
                ext = self.extension(name)
            except (ImportError, CoreError) as exc:
                log.debug("extension %s unavailable: %s", name, exc)
                continue
            provider = getattr(ext, "tools_for_role", None)
            if provider is not None:
                tools.extend(provider(role))
        return tools

    def tool_services(self, project: Project, profile: dict[str, Any]) -> ToolServices:
        return ToolServices(
            config=self.config,
            db=self.db,
            policy=self.policy,
            approvals=self.approvals,
            tasks=self.tasks,
            processes=self.processes,
            artifacts=self.artifacts,
            evidence=self.evidence,
            events=self.events,
            redactor=self.redactor,
            workspaces=self.workspaces,
            approval_handler=self.approval_handler,
            input_handler=self.input_handler,
            intel=self.index,
            memory=self.memory if self.config.memory.enabled else None,
            skills=self.skills if self.config.skills.enabled else None,
            mcp=self._extensions.get("mcp"),
            browser=self._extensions.get("browser"),
            research=self._extensions.get("research"),
            lsp=self._extensions.get("lsp"),
            profile=profile,
            base_env=self.env,
        )

    def _active_heuristics(self, kind: str) -> list[dict[str, Any]]:
        from coremain.util.jsonutil import loads

        rows = self.db.query(
            "SELECT name, version, content_json FROM heuristics WHERE kind = ? AND status = 'active'", (kind,)
        )
        return [{"name": f"{r['name']}@v{r['version']}", **loads(r["content_json"], {})} for r in rows]

    # ------------------------------------------------------------ project intel
    async def ensure_index(self, *, full: bool = False) -> dict[str, Any]:
        project = self.require_project()
        report = await self.index.update_async(project.id, Path(project.root_path), full=full)
        self.events.emit("index.updated", project_id=project.id, data=report.to_dict())
        return report.to_dict()

    async def profile_for(self, project: Project, *, refresh: bool = False) -> dict[str, Any]:
        root = Path(project.root_path)
        current_hash = await asyncio.to_thread(manifests_hash, root)
        fresh = self.projects.get(project.id)
        index_state = self.index.status(project.id)
        if not refresh and fresh.profile and fresh.profile_hash == current_hash and index_state["indexed"]:
            await self.index.update_async(project.id, root)
            return fresh.profile
        await self.index.update_async(project.id, root, full=refresh)
        files = [f["path"] for f in self.index.files(project.id)]
        prof = await asyncio.to_thread(detect_profile, root, self.index.languages(project.id), files)
        data = prof.to_dict()
        data["summary"] = prof.summary()
        self.projects.update_profile(project.id, data, current_hash)
        self.events.emit(
            "project.profiled",
            project_id=project.id,
            data={"ecosystems": prof.ecosystems, "commands": prof.commands},
        )
        return data

    def workspace_env(self, ws: Workspace, profile: dict[str, Any]) -> dict[str, str]:
        env: dict[str, str] = {}
        if ws.isolated and "python" in (profile.get("ecosystems") or []):
            src_dirs = (profile.get("python") or {}).get("src_dirs") or []
            env["PYTHONPATH"] = os.pathsep.join([*(str(ws.path / d) for d in src_dirs), str(ws.path)])
        return env

    # --------------------------------------------------------------- requests
    def ensure_session(self, session_id: str | None = None, *, new: bool = False) -> Any:
        project = self.require_project()
        if session_id:
            return self.sessions.resolve(session_id, project_id=project.id)
        if not new:
            latest = self.sessions.latest(project.id)
            if latest is not None:
                return latest
        return self.sessions.create(project.id)

    async def submit(
        self,
        text: str,
        *,
        session_id: str | None = None,
        mode: str | None = None,
        model: str | None = None,
        workspace_mode: str | None = None,
        contract: dict[str, Any] | None = None,
        auto_apply: bool | None = None,
        pins: dict[str, str] | None = None,
        title: str | None = None,
    ) -> Task:
        project = self.require_project()
        text = text.strip()
        if not text:
            raise UsageError("empty request")
        if model is not None:
            self.registry.model(model)
        session = self.ensure_session(session_id)
        self.sessions.add_message(session.id, "user", text)
        profile = await self.profile_for(project)
        index_state = self.index.status(project.id)
        cls = classify(
            text,
            repo_files=int(index_state.get("files") or 0),
            mode_override=mode,
            models_available=len(self.registry.models()),
            has_tests=bool(
                (profile.get("commands") or {}).get("test") or self.config.verification.commands.get("test")
            ),
        )
        merged_contract = self._default_contract()
        for key, value in (contract or {}).items():
            if isinstance(value, list):
                merged_contract[key] = [*merged_contract.get(key, []), *value]
            else:
                merged_contract[key] = value
        reqs = plan_requirements(
            changes_code=cls.changes_code,
            kind=cls.kind,
            profile=profile,
            contract=merged_contract,
            config=self.config.verification,
        )
        merged_contract["requirements"] = requirements_to_dicts(reqs)
        options: dict[str, Any] = {}
        if model:
            options["model"] = model
        if pins:
            options["pins"] = pins
        if workspace_mode:
            options["workspace"] = workspace_mode
        if auto_apply is not None:
            options["auto_apply"] = auto_apply
        task = self.tasks.create(
            project_id=project.id,
            session_id=session.id,
            title=title or one_line(text, 90),
            description=text,
            kind=cls.kind,
            contract=merged_contract,
            options=options,
            mode=cls.mode,
        )
        decision = {"classification": cls.to_dict(), "workflow": cls.workflow, "reasons": cls.reasons}
        task = self.tasks.update_fields(task.id, decision=decision)
        self.events.emit(
            "mode.selected",
            project_id=project.id,
            session_id=session.id,
            task_id=task.id,
            data={
                "mode": cls.mode,
                "workflow": cls.workflow,
                "kind": cls.kind,
                "risk": cls.risk,
                "reasons": cls.reasons,
                "requirements": [r.description or r.kind for r in reqs if r.required],
                "review_strategies": cls.review_strategies,
            },
        )
        return task

    def _default_contract(self) -> dict[str, Any]:
        if self.project_root is None:
            return {}
        path = ProjectPaths(self.project_root).contracts_dir / "default.toml"
        if not path.exists():
            return {}
        from coremain.config.loader import read_toml

        data = read_toml(path)
        allowed = {
            "acceptance_criteria",
            "constraints",
            "forbidden_paths",
            "required_tests",
            "required_commands",
            "required_checks",
            "browser_checks",
            "focus_paths",
        }
        return {k: v for k, v in data.items() if k in allowed}

    # --------------------------------------------------------------- execution
    async def run_task(self, task_id: str, *, nested: bool = False) -> Task:
        from coremain.runtime.runner import TaskRunner

        if self._sem is None:
            self._sem = asyncio.Semaphore(self.config.budgets.max_workers)
        token = self._cancels.get(task_id) or CancelToken()
        self._cancels[task_id] = token
        self._worker_seq += 1
        worker = f"w{self._worker_seq}"
        try:
            if nested:
                final = await TaskRunner(self, worker).execute(task_id, token)
            else:
                async with self._sem:
                    final = await TaskRunner(self, worker).execute(task_id, token)
            if final.status == TaskStatus.COMPLETED:
                self.tasks.release_dependents(task_id)
            return final
        finally:
            self._cancels.pop(task_id, None)

    def start_task(self, task_id: str) -> asyncio.Task[Task]:
        existing = self._running.get(task_id)
        if existing is not None and not existing.done():
            return existing
        t = asyncio.create_task(self.run_task(task_id))
        self._running[task_id] = t

        def forget(_t: asyncio.Task[Task], tid: str = task_id) -> None:
            self._running.pop(tid, None)

        t.add_done_callback(forget)
        return t

    def cancel_task(self, task_id: str, *, reason: str = "cancelled by user") -> Task:
        task = self.tasks.request_cancel(task_id, reason=reason)
        token = self._cancels.get(task_id)
        if token is not None:
            token.cancel(reason)
        for child in self.tasks.list(parent_task_id=task_id, limit=200):
            child_token = self._cancels.get(child.id)
            if child_token is not None:
                child_token.cancel(reason)
        return task

    def answer(self, task_id: str, text: str) -> Task:
        pending = [a for a in self.approvals.pending(task_id=task_id) if a.capability == "user_input"]
        if not pending:
            raise NotFoundError(f"task {task_id} has no pending question")
        self.approvals.decide(pending[-1].id, True, reason=text)
        task = self.tasks.get(task_id)
        if task.status == TaskStatus.NEEDS_INPUT:
            task = self.tasks.resume(task_id, note="user answered the question")
        return task

    def decide_approval(
        self, approval_id: str, approved: bool, *, scope: str = "once", reason: str | None = None
    ) -> Any:
        approval = self.approvals.decide(approval_id, approved, scope=scope, reason=reason)  # type: ignore[arg-type]
        if approval.task_id:
            task = self.tasks.get(approval.task_id)
            live = approval.task_id in self._cancels
            if task.status == TaskStatus.AWAITING_APPROVAL and not live:
                attempts = self.tasks.attempts(task.id)
                if attempts and attempts[-1].status.value == "suspended":
                    self.tasks.resume(task.id, note=f"approval {approval.status}")
        return approval

    # ----------------------------------------------------------------- context
    def recent_conversation(self, task: Task, *, limit: int = 4) -> list[str]:
        if not task.session_id:
            return []
        messages = [
            m for m in self.sessions.messages(task.session_id, limit=limit * 3) if m.task_id != task.id
        ]
        out = []
        for m in messages[-limit:]:
            if m.role == "user" and m.content.strip() == task.description.strip():
                continue
            out.append(f"{m.role}: {one_line(m.content, 700)}")
        return out

    def record_context_snapshot(
        self, task_id: str, attempt_id: str, stage: str, compiled: CompiledContext
    ) -> str:
        now = self.clock.now()
        snap_id = new_id("ctx", now=now)
        content_id = None
        if self.config.context.snapshot_content:
            art = self.artifacts.put_text(
                compiled.render(),
                kind="context",
                name=stage,
                project_id=self.project.id if self.project else None,
                task_id=task_id,
                attempt_id=attempt_id,
                media_type="text/markdown",
            )
            content_id = art.id
        self.db.execute(
            "INSERT INTO context_snapshots(id, task_id, attempt_id, stage, strategy, strategy_version, cache_key, budget_tokens, used_tokens, "
            "manifest_json, content_artifact_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                snap_id,
                task_id,
                attempt_id,
                stage,
                compiled.strategy,
                compiled.strategy_version,
                compiled.cache_key,
                compiled.budget,
                compiled.used,
                dumps(compiled.manifest()),
                content_id,
                now,
            ),
        )
        return snap_id

    def learn_from_success(self, run: Any, gate: Any) -> None:
        if not self.config.learning.enabled or self.project is None:
            return
        for ev in self.evidence.for_task(run.task.id, attempt_id=run.attempt.id, kinds=(EvidenceKind.TESTS,)):
            if ev.status == "pass" and ev.command and ev.data.get("name") == "suite":
                self.memory.add(
                    content=f"The project's test suite runs and passes with `{ev.command}`.",
                    kind="fact",
                    scope="operational",
                    source_type="evidence",
                    project_id=self.project.id,
                    evidence_ids=[ev.id],
                    tags=["tests", "commands"],
                    source_ref=f"task:{run.task.id}",
                )
                break

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": __version__,
            "runtime_id": self.runtime_id,
            "data_dir": str(self.paths.data_dir),
            "config_warnings": self.effective.warnings,
        }
        out["providers"] = [
            {
                "id": pid,
                "kind": cfg.kind,
                "enabled": cfg.enabled,
                "credential": self.registry.credential_status(pid).describe(),
            }
            for pid, cfg in self.config.providers.items()
        ]
        out["models"] = [{"key": m.key, "ref": m.ref, "tier": m.config.tier} for m in self.registry.models()]
        out["permissions"] = self.policy.describe()
        if self.project is not None:
            project = self.projects.get(self.project.id)
            counts = {
                r["status"]: r["n"]
                for r in self.db.query(
                    "SELECT status, COUNT(*) AS n FROM tasks WHERE project_id = ? GROUP BY status",
                    (project.id,),
                )
            }
            attention = self.tasks.list(project_id=project.id, statuses=ATTENTION, limit=20)
            active = self.tasks.list(
                project_id=project.id,
                statuses={TaskStatus.RUNNING, TaskStatus.VERIFYING, TaskStatus.REVIEWING, TaskStatus.QUEUED},
                limit=20,
            )
            session = self.sessions.latest(project.id)
            out["project"] = {
                "id": project.id,
                "root": project.root_path,
                "name": project.name,
                "vcs": project.vcs,
                "trusted": self.project_trusted,
                "profile": (project.profile or {}).get("summary"),
            }
            out["session"] = {"id": session.id, "title": session.title} if session else None
            out["tasks"] = {
                "counts": counts,
                "active": [self._task_brief(t) for t in active],
                "attention": [self._task_brief(t) for t in attention],
            }
            out["index"] = self.index.status(project.id)
            out["approvals"] = [a.to_dict() for a in self.approvals.pending(project_id=project.id)]
        return out

    @staticmethod
    def _task_brief(t: Task) -> dict[str, Any]:
        return {
            "id": t.id,
            "title": t.title,
            "status": t.status.value,
            "mode": t.mode,
            "kind": t.kind,
            "reason": t.status_reason,
            "block_reason": t.block_reason,
            "updated_at": t.updated_at,
        }
