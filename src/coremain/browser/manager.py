"""BrowserManager: the browser extension of CoreRuntime.

Owns the lazily started Playwright driver and Chromium process, one isolated session per task
(plus ephemeral sessions for checks, CLI commands and live probes), the model-facing tools and
diagnostics. Nothing is launched until a session is actually needed; when the last session
closes the browser is shut down again, so an idle runtime holds no browser processes.

Chromium is launched with a placeholder global proxy that every context overrides with its
own egress proxy; a context created without one therefore has no network access at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import importlib.util
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coremain.browser.origins import (
    ALLOW_HINT,
    FileScope,
    OriginPolicy,
    Verdict,
    check_target,
    file_url_to_path,
    policy_target,
    resolve_target,
)
from coremain.config.schema import BrowserConfig
from coremain.errors import CapabilityUnavailable, CoreError, PolicyDeniedError, ToolError, UsageError
from coremain.security.env import build_subprocess_env
from coremain.security.paths import is_within
from coremain.security.policy import Capability, PolicyRequest
from coremain.util.ids import new_id
from coremain.util.jsonutil import sha256_hex

if TYPE_CHECKING:
    from coremain.browser.session import BrowserSession
    from coremain.runtime.app import CoreRuntime
    from coremain.runtime.cancel import CancelToken
    from coremain.tools.base import Tool, ToolContext

log = logging.getLogger(__name__)

BROWSER_ROLES = frozenset({"implementer", "corrector", "debugger", "reviewer", "researcher"})
MAX_SESSIONS = 8
DEFAULT_VIEWPORT = (1280, 800)
LAUNCH_TIMEOUT_MS = 60_000
BLOCK_EVENT_LIMIT = 10
INSTALL_HINT = (
    "install the browser extra (`pip install 'core-main[browser]'`, or `uv sync` in a development checkout), "
    "then download Chromium with `playwright install chromium`"
)
# WebRTC may only use proxied transports (and so the egress policy); QUIC cannot be proxied.
LAUNCH_ARGS = ("--force-webrtc-ip-handling-policy=disable_non_proxied_udp", "--disable-quic")


@lru_cache(maxsize=1)
def playwright_version() -> str | None:
    """Installed Playwright version, or ``None`` when the package is not importable."""
    if importlib.util.find_spec("playwright") is None:
        return None
    try:
        return importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def session_key(ctx: ToolContext) -> str:
    return ctx.task_id or f"workspace:{ctx.workspace.id}"


def classify_launch_error(text: str, *, headless: bool) -> tuple[str, str]:
    stripped = text.strip()
    first = stripped.splitlines()[0][:300] if stripped else "unknown error"
    lowered = stripped.lower()
    if "executable doesn't exist" in lowered or "executable does not exist" in lowered:
        return (
            f"the Chromium build Playwright expects is not installed ({first})",
            "run `playwright install chromium` (in a uv checkout: `uv run playwright install chromium`) "
            "or set browser.executable_path",
        )
    if "shared librar" in lowered or "missing dependencies" in lowered or "install-deps" in lowered:
        return (
            "Chromium cannot start because system libraries are missing",
            "run `playwright install-deps chromium` as root, or install the libraries Playwright lists",
        )
    if not headless and ("display" in lowered or "x server" in lowered):
        return (
            "headed Chromium needs a display",
            "set browser.headless = true or run inside a desktop session",
        )
    if "timeout" in lowered:
        return (
            f"Chromium did not start within {LAUNCH_TIMEOUT_MS // 1000}s",
            "check system load and `core browser check --live`",
        )
    return (f"Chromium failed to start: {first}", "run `core browser check --live` for details")


def _is_executable(path: str) -> bool:
    p = Path(path).expanduser()
    return p.is_file() and os.access(p, os.X_OK)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class BrowserManager:
    def __init__(self, rt: CoreRuntime):
        self.rt = rt
        self.origins = OriginPolicy.from_config(rt.config)
        self._pw: Any = None
        self._browser: Any = None
        self._launch: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._sessions: OrderedDict[str, BrowserSession] = OrderedDict()
        self._ephemeral: set[BrowserSession] = set()
        self._holds = 0
        self._closing = False

    @property
    def config(self) -> BrowserConfig:
        return self.rt.config.browser

    # ------------------------------------------------------------ availability
    def unavailable_reason(self) -> str | None:
        cfg = self.config
        if not cfg.enabled:
            return "browser automation is disabled (browser.enabled = false)"
        if playwright_version() is None:
            return f"Playwright is not installed; {INSTALL_HINT}"
        if cfg.executable_path and not _is_executable(cfg.executable_path):
            return f"browser.executable_path {cfg.executable_path!r} does not exist or is not executable"
        return None

    def tools_for_role(self, role: str) -> list[Tool]:
        if role not in BROWSER_ROLES or not self.config.enabled or playwright_version() is None:
            return []
        from coremain.browser.tools import browser_tools

        return browser_tools(self)

    async def check(self, *, live: bool = False) -> dict[str, Any]:
        cfg = self.config
        out: dict[str, Any] = {
            "enabled": cfg.enabled,
            "playwright": playwright_version(),
            "headless": cfg.headless,
            "executable_path": cfg.executable_path,
            "allowed_origins": list(cfg.allowed_origins),
            "invalid_origins": list(self.origins.invalid),
            "offline": self.origins.offline,
            "sessions": len(self._sessions),
            "live": live,
        }
        problem = self.unavailable_reason()
        if problem is not None:
            status = "disabled" if not cfg.enabled else "unavailable"
            return {**out, "status": status, "detail": problem}
        if not live:
            detail = f"Playwright {out['playwright']} is installed; Chromium was not launched (use --live to verify)"
        else:
            try:
                info = await self._probe()
            except CapabilityUnavailable as exc:
                return {
                    **out,
                    "status": "unavailable",
                    "detail": exc.message + (f"; {exc.hint}" if exc.hint else ""),
                }
            out.update(info)
            detail = f"launched Chromium {info['browser_version']} and captured a snapshot in {info['probe_ms']} ms"
        if self.origins.invalid:
            detail += f"; ignored invalid browser.allowed_origins entries: {', '.join(self.origins.invalid)}"
        return {**out, "status": "ready" if live else "available", "detail": detail}

    async def _probe(self) -> dict[str, Any]:
        started = time.monotonic()
        async with self.ephemeral(FileScope(()), purpose="probe") as session, session.lock:
            page = session.require_page()
            await page.set_content("<main><h1>Core Main browser check</h1><button>OK</button></main>")
            snap = await session.snapshot()
        if "Core Main browser check" not in snap.text:
            raise CapabilityUnavailable(
                "Chromium started but did not produce a usable accessibility snapshot"
            )
        return {
            "browser_version": self._launch.get("version"),
            "launch_ms": self._launch.get("launch_ms"),
            "probe_ms": int((time.monotonic() - started) * 1000),
        }

    # --------------------------------------------------------------- lifecycle
    def _emit(
        self,
        kind: str,
        *,
        level: str = "info",
        data: dict[str, Any] | None = None,
        project_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        try:
            self.rt.events.emit(kind, level=level, project_id=project_id, task_id=task_id, data=data or {})
        except (sqlite3.Error, CoreError) as exc:
            log.debug("could not record %s: %s", kind, exc)

    async def _ensure_browser(self) -> Any:
        if self._browser is not None and self._browser.is_connected():
            return self._browser
        self._browser = None
        problem = self.unavailable_reason()
        if problem is not None:
            raise CapabilityUnavailable(problem, hint=INSTALL_HINT if "not installed" in problem else None)
        try:
            from playwright.async_api import Error as PlaywrightError
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise CapabilityUnavailable(f"Playwright cannot be imported: {exc}", hint=INSTALL_HINT) from exc
        cfg = self.config
        started = time.monotonic()
        try:
            if self._pw is None:
                self._pw = await async_playwright().start()
            browser = await self._pw.chromium.launch(
                headless=cfg.headless,
                executable_path=cfg.executable_path or None,
                args=[*LAUNCH_ARGS, *cfg.launch_args],
                proxy={"server": "http://per-context"},
                env=build_subprocess_env(self.rt.env, passthrough=self.rt.config.permissions.env_passthrough),
                timeout=LAUNCH_TIMEOUT_MS,
            )
        except PlaywrightError as exc:
            message, hint = classify_launch_error(
                getattr(exc, "message", "") or str(exc), headless=cfg.headless
            )
            await self._stop_driver()  # a half-started driver is restarted on the next attempt
            self._emit("browser.unavailable", level="warning", data={"reason": message, "hint": hint})
            raise CapabilityUnavailable(message, hint=hint) from exc
        browser.on("disconnected", self._on_disconnected)
        self._browser = browser
        self._launch = {
            "version": browser.version,
            "headless": cfg.headless,
            "launch_ms": int((time.monotonic() - started) * 1000),
        }
        self._emit("browser.launched", data={"browser": "chromium", **self._launch})
        return browser

    def _on_disconnected(self, browser: Any) -> None:
        if browser is self._browser:
            self._browser = None
        if self._closing:
            return
        affected = [*self._sessions.values(), *self._ephemeral]
        for session in affected:
            session.mark_lost("the browser process disconnected")
        self._emit("browser.disconnected", level="warning", data={"sessions": len(affected)})

    def _on_block(self, session: BrowserSession, target: str, reason: str, layer: str) -> None:
        if session.blocked_total <= BLOCK_EVENT_LIMIT:
            self._emit(
                "browser.blocked",
                level="warning",
                project_id=session.project_id,
                task_id=session.task_id,
                data={"target": target, "reason": reason, "layer": layer, "session": session.key},
            )

    async def _open(
        self,
        key: str,
        files: FileScope,
        *,
        workspace: Path | None,
        viewport: tuple[int, int] = DEFAULT_VIEWPORT,
        project_id: str | None = None,
        task_id: str | None = None,
    ) -> BrowserSession:
        from playwright.async_api import Error as PlaywrightError

        from coremain.browser.session import BrowserSession, first_line

        browser = await self._ensure_browser()
        try:
            return await BrowserSession.open(
                browser,
                key=key,
                origins=self.origins,
                files=files,
                workspace=workspace,
                navigation_timeout_s=self.config.navigation_timeout_s,
                viewport=viewport,
                on_block=self._on_block,
                project_id=project_id,
                task_id=task_id,
            )
        except PlaywrightError as exc:
            raise CapabilityUnavailable(
                f"could not open a browser context: {first_line(exc)}", hint="run `core browser check --live`"
            ) from exc

    async def _stop_driver(self) -> None:
        pw, self._pw = self._pw, None
        if pw is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pw.stop(), timeout=15)

    async def _shutdown(self) -> None:
        self._closing = True
        try:
            browser, self._browser = self._browser, None
            if browser is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(browser.close(), timeout=15)
            await self._stop_driver()
        finally:
            self._closing = False

    async def _maybe_shutdown(self) -> None:
        if not self._sessions and not self._ephemeral and self._holds == 0 and (self._browser or self._pw):
            await self._shutdown()

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        """Keep the browser running across several ephemeral sessions (e.g. a batch of checks)."""
        self._holds += 1
        try:
            yield
        finally:
            self._holds -= 1
            await asyncio.shield(self._release(None))

    async def _release(self, session: BrowserSession | None) -> None:
        async with self._lock:
            if session is not None:
                self._ephemeral.discard(session)
                await session.close("finished")
            await self._maybe_shutdown()

    @contextlib.asynccontextmanager
    async def ephemeral(
        self,
        files: FileScope,
        *,
        purpose: str,
        viewport: tuple[int, int] = DEFAULT_VIEWPORT,
        project_id: str | None = None,
        task_id: str | None = None,
    ) -> AsyncIterator[BrowserSession]:
        """A fresh isolated session that is always closed afterwards (checks, CLI, probes)."""
        async with self._lock:
            session = await self._open(
                f"{purpose}:{new_id('bws')}",
                files,
                workspace=None,
                viewport=viewport,
                project_id=project_id,
                task_id=task_id,
            )
            self._ephemeral.add(session)
        try:
            yield session
        finally:
            await asyncio.shield(self._release(session))

    # ----------------------------------------------------------- task sessions
    def workspace_files(self, root: Path) -> FileScope:
        return FileScope.of(root, extra_sensitive=self.rt.config.permissions.sensitive_paths)

    def preflight(self, ctx: ToolContext, url: str) -> Verdict:
        """Verdict for a tool navigation, evaluated before any browser is started."""
        return check_target(self.origins, self.workspace_files(Path(ctx.workspace.path)), url)

    async def session_for(self, ctx: ToolContext, *, create: bool) -> BrowserSession | None:
        """The task's session. A closed session is returned as-is unless ``create`` replaces it;
        a session bound to a different workspace (a new attempt) is never reused."""
        key = session_key(ctx)
        root = Path(ctx.workspace.path).resolve()
        async with self._lock:
            session = self._sessions.get(key)
            if session is not None and (session.workspace != root or (create and session.closed)):
                self._sessions.pop(key)
                reason = "workspace changed" if session.workspace != root else "replaced a lost session"
                await session.close(reason)
                session = None
            if session is None:
                if not create:
                    return None
                while len(self._sessions) >= MAX_SESSIONS:
                    old_key, old = self._sessions.popitem(last=False)
                    await old.close("evicted: too many concurrent browser sessions")
                    self._emit("browser.session_evicted", data={"session": old_key}, task_id=old.task_id)
                session = await self._open(
                    key,
                    self.workspace_files(root),
                    workspace=root,
                    project_id=ctx.project_id,
                    task_id=ctx.task_id,
                )
                self._sessions[key] = session
            else:
                self._sessions.move_to_end(key)
            return session

    def current_target(self, ctx: ToolContext) -> str:
        """Policy target (origin or file URL) of the page the task's session is showing."""
        session = self._sessions.get(session_key(ctx))
        if session is None or session.closed or session.workspace != Path(ctx.workspace.path).resolve():
            return ""
        page = session.page
        if page is None or page.is_closed():
            return ""
        return policy_target(page.url)

    async def close_task(self, task_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(task_id, None)
            if session is None:
                return False
            await session.close("task finished")
            self._emit(
                "browser.session_closed",
                task_id=task_id,
                project_id=session.project_id,
                data={
                    "session": session.key,
                    "blocked": session.blocked_total,
                    "screenshots": session.screenshots,
                },
            )
            await self._maybe_shutdown()
            return True

    async def aclose(self) -> None:
        async with self._lock:
            sessions = [*self._sessions.values(), *self._ephemeral]
            self._sessions.clear()
            self._ephemeral.clear()
            for session in sessions:
                await session.close("runtime shutting down")
            await self._shutdown()

    # ---------------------------------------------------------------- checks
    async def run_checks(
        self,
        checks: list[dict[str, Any]],
        *,
        workspace_path: Path,
        task_id: str | None,
        attempt_id: str | None = None,
        cancel: CancelToken | None = None,
    ) -> list[dict[str, Any]]:
        from coremain.browser.checks import run_checks

        return await run_checks(
            self, checks, workspace_path=workspace_path, task_id=task_id, attempt_id=attempt_id, cancel=cancel
        )

    # ------------------------------------------------------------------ CLI
    def _cli_files(self, target: str) -> FileScope:
        path = file_url_to_path(target)
        extra = self.rt.config.permissions.sensitive_paths
        if path is None:
            return FileScope((), tuple(extra))
        resolved = path.resolve()
        root = self.rt.project_root
        if root is not None and is_within(root, resolved):
            return FileScope.of(root, extra_sensitive=extra)
        return FileScope.of(resolved if resolved.is_dir() else resolved.parent, extra_sensitive=extra)

    async def screenshot_url(
        self, url: str, output_path: Path, *, width: int = 1280, height: int = 800, full_page: bool = False
    ) -> dict[str, Any]:
        """User-initiated screenshot (``core browser screenshot``).

        Relative paths resolve against the current directory. The policy engine is consulted
        ("ask" counts as consent because the user issued the command; "deny" always refuses)
        and the origin allowlist, offline mode and file confinement apply as for tools.
        """
        problem = self.unavailable_reason()
        if problem is not None:
            raise CapabilityUnavailable(problem)
        if not (100 <= width <= 8000 and 100 <= height <= 8000):
            raise UsageError("width and height must be between 100 and 8000 pixels")
        cwd = Path.cwd()
        target = resolve_target(url, cwd, confine=False)
        files = self._cli_files(target)
        project_id = self.rt.project.id if self.rt.project is not None else None
        target_key = policy_target(target)
        decision = self.rt.policy.evaluate(
            PolicyRequest(
                capability=Capability.BROWSER,
                target=target_key,
                workspace_root=self.rt.project_root,
                project_id=project_id,
                tool="core browser screenshot",
            )
        )
        if decision.decision == "deny":
            raise PolicyDeniedError(f"browser access to {target_key or target} is denied: {decision.reason}")
        verdict = check_target(self.origins, files, target)
        if not verdict.allowed:
            raise PolicyDeniedError(f"{verdict.target}: {verdict.reason}", hint=ALLOW_HINT)
        out = output_path.expanduser()
        out = out if out.is_absolute() else cwd / out
        async with (
            self.ephemeral(files, purpose="cli", viewport=(width, height), project_id=project_id) as session,
            session.lock,
        ):
            nav = await session.navigate(target)
            if nav.blocked:
                raise PolicyDeniedError(f"{url} redirected to a blocked origin ({nav.url}): {nav.blocked}")
            if nav.proxy_error:
                raise ToolError(f"could not load {url}: {nav.proxy_error}", error_class="navigation_failed")
            shot = await session.screenshot(full_page=full_page)
            errors = [e.render() for e in session.errors_since(0)]
            blocked = [e.text for e in session.entries_since(0, frozenset({"blocked"}))]
        await asyncio.to_thread(_atomic_write, out, shot.data)
        info: dict[str, Any] = {
            "url": target,
            "final_url": nav.url,
            "title": nav.title,
            "http_status": nav.status,
            "path": str(out),
            "width": shot.width,
            "height": shot.height,
            "bytes": len(shot.data),
            "sha256": sha256_hex(shot.data),
            "full_page": full_page,
            "console_errors": errors[:20],
            "blocked": blocked[:20],
            "policy": decision.decision,
            "note": shot.note or nav.note,
        }
        redacted: dict[str, Any] = self.rt.redactor.redact_obj(info)
        return redacted
