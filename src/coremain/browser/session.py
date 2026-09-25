"""One browser session: a Playwright context behind its own egress proxy, plus an observation log.

Sessions are created by :class:`~coremain.browser.manager.BrowserManager` (one per task, or
ephemeral for checks and CLI commands). Every request the context makes passes the route
handler (origin policy + workspace file confinement) and every network connection passes the
session's :class:`~coremain.browser.egress.EgressProxy`. Console output, page errors, failed
and blocked requests, dialogs and popups are recorded with sequence numbers so tools and
checks can report exactly what happened since a given point.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import struct
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from coremain.browser.egress import BLOCKED_HEADER, ERROR_HEADER, EgressProxy, EgressRecord
from coremain.browser.origins import ALLOW_HINT, FileScope, OriginPolicy, Verdict, check_target
from coremain.errors import PolicyDeniedError, ToolError

if TYPE_CHECKING:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        ConsoleMessage,
        Dialog,
        Locator,
        Page,
        Request,
        Response,
        Route,
    )

REF_RE = re.compile(r"^(?:f\d+)?e\d+$")
ACTION_TIMEOUT_MS = 10_000
SNAPSHOT_TIMEOUT_MS = 10_000
SCREENSHOT_TIMEOUT_MS = 20_000
SETTLE_NAVIGATION_S = 5.0
SETTLE_ACTION_S = 3.0
QUIET_S = 0.2
SNAPSHOT_MAX_CHARS = 12_000
LOG_MAX = 500
TEXT_MAX = 500
SCREENSHOT_MAX_EDGE = 7_800
SCREENSHOT_MAX_BYTES = 4_500_000
ERROR_KINDS = frozenset({"console.error", "pageerror", "crash"})
_LONG_LIVED = frozenset({"eventsource", "websocket"})
_IGNORED_FAILURES = ("ERR_BLOCKED_BY_CLIENT", "ERR_ABORTED")

BlockCallback = Callable[["BrowserSession", str, str, str], None]


@dataclass
class LogEntry:
    seq: int
    kind: str
    text: str
    url: str | None = None
    ts: float = field(default_factory=time.time)

    @property
    def is_error(self) -> bool:
        return self.kind in ERROR_KINDS

    def render(self) -> str:
        where = f" ({self.url})" if self.url else ""
        return f"[{self.kind}] {self.text}{where}"


@dataclass
class Navigation:
    requested: str
    url: str
    status: int | None
    title: str
    note: str | None = None
    blocked: str | None = None
    proxy_error: str | None = None


@dataclass
class Snapshot:
    url: str
    title: str
    text: str
    omitted_lines: int = 0


@dataclass
class Shot:
    data: bytes
    width: int
    height: int
    url: str
    note: str | None = None


def first_line(exc: BaseException, limit: int = 300) -> str:
    text = (getattr(exc, "message", None) or str(exc) or type(exc).__name__).strip()
    return text.splitlines()[0][:limit] if text else type(exc).__name__


def call_log_hint(exc: BaseException) -> str:
    """The most informative lines of a Playwright call log (why an action could not proceed)."""
    text = getattr(exc, "message", None) or str(exc)
    _, _, log = text.partition("Call log:")
    lines: list[str] = []
    for raw in log.splitlines():
        line = raw.strip(" -\t")
        if line and line not in lines:
            lines.append(line)
    return "; ".join(lines[-3:])[:400] if lines else first_line(exc)


def bound_lines(text: str, max_chars: int) -> tuple[str, int]:
    if len(text) <= max_chars:
        return text, 0
    cut = text.rfind("\n", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    omitted = text.count("\n", cut) + 1
    return text[:cut], omitted


def png_size(data: bytes) -> tuple[int, int]:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    return 0, 0


def _clip(text: str, limit: int = TEXT_MAX) -> str:
    text = " ".join(text.split()) if "\n" in text else text
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _navigating(exc: BaseException) -> bool:
    text = first_line(exc).lower()
    return "context was destroyed" in text or "navigat" in text or "frame was detached" in text


class BrowserSession:
    def __init__(
        self,
        *,
        key: str,
        context: BrowserContext,
        proxy: EgressProxy,
        origins: OriginPolicy,
        files: FileScope,
        workspace: Path | None,
        navigation_timeout_s: float,
        on_block: BlockCallback | None = None,
        project_id: str | None = None,
        task_id: str | None = None,
    ):
        self.key = key
        self.context = context
        self.proxy = proxy
        self.origins = origins
        self.files = files
        self.workspace = workspace
        self.project_id = project_id
        self.task_id = task_id
        self.navigation_timeout_ms = int(navigation_timeout_s * 1000)
        self.lock = asyncio.Lock()
        self.page: Page | None = None
        self.closed = False
        self.close_reason: str | None = None
        self.log: deque[LogEntry] = deque(maxlen=LOG_MAX)
        self.console_cursor = 0
        self.blocked_total = 0
        self.screenshots = 0
        self._seq = 0
        self._pending: set[Request] = set()
        self._attached: set[int] = set()
        self._crashed: set[int] = set()
        self._on_block = on_block

    # ----------------------------------------------------------------- lifecycle
    @classmethod
    async def open(
        cls,
        browser: Browser,
        *,
        key: str,
        origins: OriginPolicy,
        files: FileScope,
        workspace: Path | None,
        navigation_timeout_s: float,
        viewport: tuple[int, int] = (1280, 800),
        on_block: BlockCallback | None = None,
        project_id: str | None = None,
        task_id: str | None = None,
    ) -> BrowserSession:
        proxy = EgressProxy(origins)
        await proxy.start()
        try:
            context = await browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
                proxy={"server": proxy.url},
                service_workers="block",  # requests answered by a service worker bypass routing
                accept_downloads=False,
            )
        except BaseException:
            await proxy.close()
            raise
        session = cls(
            key=key,
            context=context,
            proxy=proxy,
            origins=origins,
            files=files,
            workspace=workspace,
            navigation_timeout_s=navigation_timeout_s,
            on_block=on_block,
            project_id=project_id,
            task_id=task_id,
        )
        proxy.on_record = session._on_egress
        try:
            context.set_default_timeout(ACTION_TIMEOUT_MS)
            context.set_default_navigation_timeout(session.navigation_timeout_ms)
            context.on("page", session._on_new_page)
            context.on("close", session._on_context_close)
            await context.route("**/*", session._route)
            session._on_new_page(await context.new_page())
        except BaseException:
            await session.close("failed to initialize")
            raise
        return session

    async def close(self, reason: str = "closed") -> None:
        if self.close_reason is None:
            self.close_reason = reason
        self.closed = True
        with contextlib.suppress(PlaywrightError, TimeoutError):
            await asyncio.wait_for(self.context.close(), timeout=15)
        await self.proxy.close()

    def mark_lost(self, reason: str) -> None:
        if not self.closed:
            self.closed = True
            self.close_reason = reason
            self._add("crash", f"browser session lost: {reason}")

    # ---------------------------------------------------------------- observation
    @property
    def mark(self) -> int:
        return self._seq

    def _add(self, kind: str, text: str, url: str | None = None) -> None:
        self._seq += 1
        self.log.append(LogEntry(self._seq, kind, _clip(text), _clip(url, 300) if url else None))

    def entries_since(self, mark: int, kinds: frozenset[str] | None = None) -> list[LogEntry]:
        return [e for e in self.log if e.seq > mark and (kinds is None or e.kind in kinds)]

    def errors_since(self, mark: int) -> list[LogEntry]:
        return [e for e in self.log if e.seq > mark and e.is_error]

    def drain(self) -> tuple[list[LogEntry], int]:
        """Entries since the previous drain, and how many were lost to the bounded buffer."""
        entries = self.entries_since(self.console_cursor)
        first = entries[0].seq if entries else self._seq + 1
        dropped = max(0, first - self.console_cursor - 1)
        self.console_cursor = self._seq
        return entries, dropped

    def undrained_counts(self) -> tuple[int, int]:
        pending = self.entries_since(self.console_cursor)
        errors = sum(1 for e in pending if e.is_error)
        return errors, len(pending) - errors

    def verdict(self, url: str) -> Verdict:
        return check_target(self.origins, self.files, url)

    def _note_block(self, target: str, reason: str, layer: str, url: str | None) -> None:
        self.blocked_total += 1
        self._add("blocked", f"{target}: {reason} [{layer}]", url)
        if self._on_block is not None:
            self._on_block(self, target, reason, layer)

    def _on_egress(self, record: EgressRecord) -> None:
        if record.kind == "blocked":
            self._note_block(record.target, record.reason, "egress proxy", record.url)
        else:
            self._add("network", f"could not connect to {record.target}: {record.reason}", record.url)

    async def _route(self, route: Route) -> None:
        url = route.request.url
        verdict = self.verdict(url)
        try:
            if verdict.allowed:
                await route.fallback()
            else:
                self._note_block(verdict.target, verdict.reason, "route", url)
                await route.abort("blockedbyclient")
        except PlaywrightError:
            pass  # the page or context went away while the request was paused

    def _on_new_page(self, page: Page) -> None:
        if id(page) in self._attached:
            return
        self._attached.add(id(page))
        page.on("console", self._on_console)
        page.on("pageerror", self._on_pageerror)
        page.on("request", self._on_request)
        page.on("requestfinished", self._on_request_done)
        page.on("requestfailed", self._on_request_failed)
        page.on("dialog", self._on_dialog)
        page.on("crash", self._on_crash)
        page.on("close", self._on_page_close)
        if self.page is not None and not self.page.is_closed():
            self._add("popup", "a new tab opened and became the active page", page.url or None)
        self.page = page

    def _on_console(self, message: ConsoleMessage) -> None:
        location = message.location or {}
        where = location.get("url") or None
        if where and location.get("lineNumber") is not None:
            where = f"{where}:{location.get('lineNumber')}"
        self._add(f"console.{message.type}", message.text, where)

    def _on_pageerror(self, error: Any) -> None:
        name = getattr(error, "name", "") or ""
        message = getattr(error, "message", "") or str(error)
        stack = getattr(error, "stack", "") or ""
        frame = next((ln.strip() for ln in stack.splitlines() if ln.strip().startswith("at ")), "")
        text = f"{name}: {message}" if name and not message.startswith(name) else message
        self._add("pageerror", f"{text} {frame}".strip())

    def _on_request(self, request: Request) -> None:
        if request.resource_type not in _LONG_LIVED:
            self._pending.add(request)

    def _on_request_done(self, request: Request) -> None:
        self._pending.discard(request)

    def _on_request_failed(self, request: Request) -> None:
        self._pending.discard(request)
        failure = request.failure or ""
        if failure and not any(code in failure for code in _IGNORED_FAILURES):
            self._add("requestfailed", f"{request.method} {request.url} failed: {failure}")

    async def _on_dialog(self, dialog: Dialog) -> None:
        # beforeunload is accepted so navigation can proceed; everything else is dismissed.
        accept = dialog.type == "beforeunload"
        self._add(
            "dialog",
            f"{dialog.type}: {dialog.message} ({'accepted' if accept else 'dismissed'} automatically)",
        )
        with contextlib.suppress(PlaywrightError):
            if accept:
                await dialog.accept()
            else:
                await dialog.dismiss()

    def _on_crash(self, page: Page) -> None:
        self._crashed.add(id(page))
        self._add("crash", "the page crashed (renderer process terminated)", page.url or None)

    def _on_page_close(self, page: Page) -> None:
        self._attached.discard(id(page))
        if page is self.page:
            remaining = [p for p in self.context.pages if not p.is_closed() and p is not page]
            self.page = remaining[-1] if remaining else None

    def _on_context_close(self, _context: BrowserContext) -> None:
        if not self.closed:
            self.mark_lost("the browser context closed unexpectedly")

    # -------------------------------------------------------------------- pages
    def require_page(self) -> Page:
        if self.closed:
            raise ToolError(
                f"the browser session is gone ({self.close_reason}); call browser_navigate to start a new one",
                error_class="browser_session_lost",
            )
        page = self.page
        if page is None or page.is_closed():
            raise ToolError("no page is open; call browser_navigate first", error_class="no_page")
        if id(page) in self._crashed:
            raise ToolError(
                "the page crashed; call browser_navigate to load it again", error_class="page_crashed"
            )
        return page

    async def _navigation_page(self) -> Page:
        if self.closed:
            self.require_page()
        page = self.page
        if page is not None and id(page) in self._crashed:
            with contextlib.suppress(PlaywrightError):
                await page.close()
            page = None
        if page is None or page.is_closed() or id(page) in self._crashed:
            self.page = None
            page = await self.context.new_page()
            self._on_new_page(page)
        return page

    async def settle(self, page: Page, cap_s: float) -> str | None:
        """Wait (bounded) for the load event and for in-flight requests to finish."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + cap_s
        note = None
        try:
            await page.wait_for_load_state("load", timeout=cap_s * 1000)
        except PlaywrightTimeoutError:
            note = f"the load event did not fire within {cap_s:.0f}s; the page may still be loading"
        except PlaywrightError:
            return None
        quiet_since: float | None = None
        while loop.time() < deadline:
            if self._pending:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = loop.time()
            elif loop.time() - quiet_since >= QUIET_S:
                break
            await asyncio.sleep(0.05)
        return note

    @staticmethod
    async def title(page: Page) -> str:
        try:
            return _clip(await page.title(), 200)
        except PlaywrightError:
            return ""

    # ------------------------------------------------------------------- actions
    async def navigate(self, url: str) -> Navigation:
        verdict = self.verdict(url)
        if not verdict.allowed:
            raise PolicyDeniedError(f"{verdict.target}: {verdict.reason}", hint=ALLOW_HINT)
        page = await self._navigation_page()
        mark = self._seq
        for attempt in range(2):
            try:
                response = await page.goto(
                    url, wait_until="domcontentloaded", timeout=self.navigation_timeout_ms
                )
                break
            except PlaywrightTimeoutError as exc:
                raise ToolError(
                    f"navigation to {url} timed out after {self.navigation_timeout_ms // 1000}s",
                    error_class="timeout",
                ) from exc
            except PlaywrightError as exc:
                # After a blocked navigation Chromium commits its own chrome-error:// page
                # asynchronously; that late navigation can interrupt the next goto once.
                if (
                    attempt == 0
                    and "interrupted by another navigation" in str(exc)
                    and "chrome-error://" in str(exc)
                ):
                    with contextlib.suppress(PlaywrightError):
                        await page.wait_for_load_state("load", timeout=5000)
                    continue
                raise self._navigation_error(url, exc, mark) from exc
        note = await self.settle(page, SETTLE_NAVIGATION_S)
        return await self._navigation(url, page, response, note, mark)

    def _navigation_error(self, url: str, exc: PlaywrightError, mark: int) -> Exception:
        message = first_line(exc)
        blocks = self.entries_since(mark, frozenset({"blocked"}))
        failures = self.entries_since(mark, frozenset({"network"}))
        if "ERR_BLOCKED_BY_CLIENT" in message or (blocks and "ERR_TUNNEL_CONNECTION_FAILED" in message):
            detail = blocks[-1].text if blocks else "blocked by the browser origin policy"
            return PolicyDeniedError(f"navigation to {url} was blocked: {detail}", hint=ALLOW_HINT)
        if failures and "ERR_TUNNEL_CONNECTION_FAILED" in message:
            return ToolError(f"could not load {url}: {failures[-1].text}", error_class="navigation_failed")
        return ToolError(f"could not load {url}: {message}", error_class="navigation_failed")

    async def _navigation(
        self, requested: str, page: Page, response: Response | None, note: str | None, mark: int
    ) -> Navigation:
        headers = response.headers if response is not None else {}
        blocked = proxy_error = None
        if headers.get(BLOCKED_HEADER.lower()):
            blocks = self.entries_since(mark, frozenset({"blocked"}))
            blocked = blocks[-1].text if blocks else "blocked by the browser origin policy"
        if headers.get(ERROR_HEADER.lower()):
            failures = self.entries_since(mark, frozenset({"network"}))
            proxy_error = (
                failures[-1].text if failures else f"proxy error: {headers.get(ERROR_HEADER.lower())}"
            )
        return Navigation(
            requested=requested,
            url=page.url,
            status=response.status if response is not None else None,
            title=await self.title(page),
            note=note,
            blocked=blocked,
            proxy_error=proxy_error,
        )

    async def snapshot(self, *, max_chars: int = SNAPSHOT_MAX_CHARS) -> Snapshot:
        page = self.require_page()
        raw = ""
        for attempt in (0, 1):
            try:
                raw = await page.aria_snapshot(mode="ai", timeout=SNAPSHOT_TIMEOUT_MS)
                break
            except PlaywrightTimeoutError as exc:
                raise ToolError("the accessibility snapshot timed out", error_class="timeout") from exc
            except PlaywrightError as exc:
                if attempt == 0 and _navigating(exc):
                    with contextlib.suppress(PlaywrightError):
                        await page.wait_for_load_state("domcontentloaded", timeout=5000)
                    continue
                raise ToolError(f"snapshot failed: {first_line(exc)}", error_class="snapshot_failed") from exc
        text, omitted = bound_lines(raw, max_chars)
        return Snapshot(page.url, await self.title(page), text, omitted)

    async def locate(self, page: Page, ref: str | None, selector: str | None) -> Locator:
        if ref:
            ref = ref.strip().removeprefix("ref=").strip("[]")
            if not REF_RE.fullmatch(ref):
                raise ToolError(
                    f"'{ref}' is not a snapshot ref (expected something like 'e12' or 'f1e3')",
                    error_class="invalid_arguments",
                )
            locator, what = page.locator(f"aria-ref={ref}"), f"ref {ref}"
        elif selector:
            locator, what = page.locator(selector), f"selector {selector!r}"
        else:
            raise ToolError(
                "provide a ref from browser_snapshot or a selector", error_class="invalid_arguments"
            )
        try:
            count = await locator.count()
        except PlaywrightError as exc:
            raise ToolError(f"invalid {what}: {first_line(exc)}", error_class="invalid_selector") from exc
        if count == 0:
            hint = (
                "the page changed since the last snapshot or the ref is stale; call browser_snapshot for fresh refs"
                if ref
                else "no element matches"
            )
            raise ToolError(f"{what} not found: {hint}", error_class="element_not_found")
        if count > 1:
            raise ToolError(
                f"{what} matches {count} elements; use a ref from browser_snapshot or a more specific selector",
                error_class="ambiguous_selector",
            )
        return locator

    async def click(self, ref: str | None, selector: str | None) -> str | None:
        page = self.require_page()
        locator = await self.locate(page, ref, selector)
        try:
            await locator.click(timeout=ACTION_TIMEOUT_MS)
        except PlaywrightTimeoutError as exc:
            raise ToolError(
                f"click did not complete within {ACTION_TIMEOUT_MS // 1000}s: {call_log_hint(exc)}",
                error_class="timeout",
            ) from exc
        except PlaywrightError as exc:
            raise ToolError(f"click failed: {first_line(exc)}", error_class="action_failed") from exc
        return await self.settle(self.page or page, SETTLE_ACTION_S)

    async def fill(self, ref: str | None, selector: str | None, text: str, *, submit: bool) -> str | None:
        page = self.require_page()
        locator = await self.locate(page, ref, selector)
        try:
            await locator.fill(text, timeout=ACTION_TIMEOUT_MS)
            if submit:
                await locator.press("Enter", timeout=ACTION_TIMEOUT_MS)
        except PlaywrightTimeoutError as exc:
            raise ToolError(
                f"typing did not complete within {ACTION_TIMEOUT_MS // 1000}s: {call_log_hint(exc)}",
                error_class="timeout",
            ) from exc
        except PlaywrightError as exc:
            raise ToolError(f"typing failed: {first_line(exc)}", error_class="action_failed") from exc
        return await self.settle(self.page or page, SETTLE_ACTION_S)

    async def screenshot(self, *, full_page: bool = False) -> Shot:
        page = self.require_page()

        async def capture(full: bool) -> bytes:
            try:
                return await page.screenshot(
                    type="png",
                    full_page=full,
                    timeout=SCREENSHOT_TIMEOUT_MS,
                    animations="disabled",
                    caret="hide",
                )
            except PlaywrightTimeoutError as exc:
                raise ToolError("the screenshot timed out", error_class="timeout") from exc
            except PlaywrightError as exc:
                raise ToolError(
                    f"screenshot failed: {first_line(exc)}", error_class="screenshot_failed"
                ) from exc

        data = await capture(full_page)
        width, height = png_size(data)
        note = None
        if full_page and (height > SCREENSHOT_MAX_EDGE or len(data) > SCREENSHOT_MAX_BYTES):
            note = f"the full page ({width}x{height}px) is too large for a model image; captured the viewport instead"
            data = await capture(False)
            width, height = png_size(data)
        self.screenshots += 1
        return Shot(data, width, height, page.url, note)
