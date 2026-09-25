"""Declarative browser verification checks (the ``browser_checks`` task-contract entries).

Each check runs in a fresh browser context (nothing carries over from the model's interactive
session) after the same target resolution, policy decision and origin verdict that tools
get. Results are plain dicts for the verifier to record as evidence:

* ``pass`` – the page loaded and every assertion held;
* ``fail`` – the page loaded but an assertion did not hold (HTTP error status, selector not
  visible, text missing, console errors, redirect to a blocked origin);
* ``error`` – the check could not be executed (invalid definition, policy refusal, browser
  unavailable, page unreachable, timeout). Errors are never reported as passes.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from coremain.browser.origins import FileScope, TargetError, check_target, policy_target, resolve_target
from coremain.errors import CapabilityUnavailable, NotFoundError, PolicyDeniedError, ToolError
from coremain.security.policy import Capability, PolicyRequest

if TYPE_CHECKING:
    from playwright.async_api import Page

    from coremain.browser.manager import BrowserManager
    from coremain.browser.session import BrowserSession, Navigation
    from coremain.runtime.cancel import CancelToken

MAX_CHECKS = 50
DEFAULT_WAIT_S = 10.0
MAX_LISTED = 20


class BrowserCheckSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=1, max_length=4000)
    expect_text: str | None = Field(None, min_length=1, max_length=2000)
    selector: str | None = Field(None, min_length=1, max_length=1000)
    expect_no_console_errors: bool = False
    screenshot: bool = False
    timeout_s: float | None = Field(None, gt=0, le=300)


class _CheckError(Exception):
    """The check definition cannot be evaluated against the page (e.g. an invalid selector)."""


def _norm(text: str) -> str:
    return " ".join(text.split())


def _excerpt(text: str, limit: int = 160) -> str:
    flat = _norm(text)
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")[:60] or "check"


def _result(name: str, status: str, summary: str, started: float, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "status": status,
        "summary": summary,
        "console_errors": [],
        "blocked": [],
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def _validate(raw: Any) -> tuple[BrowserCheckSpec | None, str | None]:
    if not isinstance(raw, dict):
        return None, "a browser check must be a table with at least 'name' and 'url'"
    try:
        return BrowserCheckSpec.model_validate(raw), None
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'check'}: {e['msg']}" for e in exc.errors()[:5]
        )
        return None, f"invalid browser check definition ({problems})"


def _project_for(manager: BrowserManager, task_id: str | None) -> str | None:
    rt = manager.rt
    if task_id:
        with contextlib.suppress(NotFoundError):
            return rt.tasks.get(task_id).project_id
    return rt.project.id if rt.project is not None else None


async def _wait_selector(page: Page, selector: str, wait_s: float) -> str | None:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    from coremain.browser.session import first_line

    try:
        await page.locator(selector).first.wait_for(state="visible", timeout=wait_s * 1000)
    except PlaywrightTimeoutError:
        return f"selector {selector!r} did not match a visible element within {wait_s:.0f}s"
    except PlaywrightError as exc:
        raise _CheckError(f"invalid selector {selector!r}: {first_line(exc)}") from exc
    return None


async def _wait_text(page: Page, selector: str | None, expected: str, wait_s: float) -> tuple[bool, str]:
    from playwright.async_api import Error as PlaywrightError

    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_s
    want = _norm(expected)
    actual = ""
    while True:
        try:
            if selector:
                actual = "\n".join(await page.locator(selector).all_inner_texts())
            else:
                actual = await page.locator("body").inner_text(timeout=2000)
        except PlaywrightError:
            actual = ""  # the document is being replaced; retry until the deadline
        if want in _norm(actual):
            return True, actual
        if loop.time() >= deadline:
            return False, actual
        await asyncio.sleep(0.25)


async def _assert(
    session: BrowserSession, spec: BrowserCheckSpec, nav: Navigation, mark: int, wait_s: float
) -> tuple[str, str]:
    if nav.blocked:
        return "fail", f"the page redirected to a blocked origin ({nav.url}): {nav.blocked}"
    if nav.proxy_error:
        return "error", f"could not load {spec.url}: {nav.proxy_error}"
    if nav.status is not None and nav.status >= 400:
        return "fail", f"{nav.url} responded with HTTP {nav.status}"
    page = session.require_page()
    passed = [f"page loaded (HTTP {nav.status})" if nav.status else "page loaded"]
    if spec.selector:
        problem = await _wait_selector(page, spec.selector, wait_s)
        if problem:
            return "fail", problem
        passed.append(f"selector {spec.selector!r} visible")
    if spec.expect_text:
        found, actual = await _wait_text(page, spec.selector, spec.expect_text, wait_s)
        if not found:
            scope = f"elements matching {spec.selector!r}" if spec.selector else "the page"
            return "fail", (
                f"expected text {spec.expect_text!r} not found in {scope} within {wait_s:.0f}s "
                f"(untrusted page text begins: {_excerpt(actual)!r})"
            )
        passed.append(f"text {spec.expect_text!r} present")
    if spec.expect_no_console_errors:
        await session.settle(page, 1.5)
        errors = session.errors_since(mark)
        if errors:
            return "fail", f"{len(errors)} console/page error(s), first: {errors[0].render()}"
        passed.append("no console errors")
    return "pass", "; ".join(passed)


async def _execute(
    manager: BrowserManager,
    spec: BrowserCheckSpec,
    target: str,
    files: FileScope,
    *,
    project_id: str | None,
    task_id: str | None,
    attempt_id: str | None,
    wait_s: float,
    started: float,
) -> dict[str, Any]:
    redact = manager.rt.redactor.redact
    nav: Navigation | None = None
    artifact_id = None
    async with (
        manager.ephemeral(files, purpose="check", project_id=project_id, task_id=task_id) as session,
        session.lock,
    ):
        mark = session.mark
        try:
            nav = await session.navigate(target)
            status, summary = await _assert(session, spec, nav, mark, wait_s)
        except PolicyDeniedError as exc:
            # The target itself was allowed (checked beforehand), so the page left the allowlist.
            status, summary = "fail", f"navigation was blocked: {exc.message}"
        except ToolError as exc:
            status, summary = "error", exc.message
        except _CheckError as exc:
            status, summary = "error", str(exc)
        if spec.screenshot and nav is not None and status != "error":
            try:
                shot = await session.screenshot(full_page=False)
            except ToolError as exc:
                summary += f"; screenshot failed: {exc.message}"
            else:
                artifact_id = manager.rt.artifacts.put_bytes(
                    shot.data,
                    kind="screenshot",
                    project_id=project_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    name=f"browser-check-{_slug(spec.name)}.png",
                    media_type="image/png",
                    meta={
                        "check": spec.name,
                        "status": status,
                        "url": redact(shot.url),
                        "width": shot.width,
                        "height": shot.height,
                    },
                ).id
        console_errors = [redact(e.render())[:300] for e in session.errors_since(mark)][:MAX_LISTED]
        blocked = [redact(e.text)[:300] for e in session.entries_since(mark, frozenset({"blocked"}))]
    return _result(
        spec.name,
        status,
        redact(summary),
        started,
        url=redact(target),
        final_url=redact(nav.url) if nav else None,
        title=redact(nav.title) if nav else None,
        http_status=nav.status if nav else None,
        artifact_id=artifact_id,
        console_errors=console_errors,
        blocked=blocked[:MAX_LISTED],
    )


async def run_checks(
    manager: BrowserManager,
    checks: list[dict[str, Any]],
    *,
    workspace_path: Path,
    task_id: str | None,
    attempt_id: str | None = None,
    cancel: CancelToken | None = None,
) -> list[dict[str, Any]]:
    rt = manager.rt
    workspace = Path(workspace_path).resolve()
    project_id = _project_for(manager, task_id)
    files = FileScope.of(workspace, extra_sensitive=rt.config.permissions.sensitive_paths)
    unavailable = manager.unavailable_reason()
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    async with manager.hold():
        for index, raw in enumerate(checks):
            if cancel is not None:
                cancel.raise_if_cancelled()
            started = time.monotonic()
            fallback = raw.get("name") if isinstance(raw, dict) and isinstance(raw.get("name"), str) else None
            name = fallback or f"check-{index + 1}"
            spec, problem = _validate(raw)
            if spec is None:
                results.append(_result(name, "error", problem or "invalid browser check", started))
                continue
            if spec.name in seen:
                results.append(_result(spec.name, "error", "duplicate browser check name", started))
                continue
            seen.add(spec.name)
            if index >= MAX_CHECKS:
                results.append(_result(spec.name, "error", f"more than {MAX_CHECKS} browser checks", started))
                continue
            if unavailable:
                results.append(_result(spec.name, "error", unavailable, started))
                continue
            try:
                target = resolve_target(spec.url, workspace)
            except (TargetError, PolicyDeniedError) as exc:
                results.append(_result(spec.name, "error", f"invalid check URL: {exc.message}", started))
                continue
            decision = rt.policy.evaluate(
                PolicyRequest(
                    capability=Capability.BROWSER,
                    target=policy_target(target),
                    workspace_root=workspace,
                    project_id=project_id,
                    task_id=task_id,
                    tool="browser_check",
                )
            )
            if decision.decision != "allow":
                advice = (
                    "; verification never waits for approvals, so allow this origin with a browser permission rule"
                    if decision.decision == "ask"
                    else ""
                )
                results.append(
                    _result(
                        spec.name, "error", f"policy {decision.decision}: {decision.reason}{advice}", started
                    )
                )
                continue
            verdict = check_target(manager.origins, files, target)
            if not verdict.allowed:
                results.append(
                    _result(spec.name, "error", f"{verdict.target}: {verdict.reason}", started, url=target)
                )
                continue
            wait_s = spec.timeout_s or DEFAULT_WAIT_S
            budget = manager.config.navigation_timeout_s + 2 * wait_s + 60
            work = _execute(
                manager,
                spec,
                target,
                files,
                project_id=project_id,
                task_id=task_id,
                attempt_id=attempt_id,
                wait_s=wait_s,
                started=started,
            )
            try:
                result = await asyncio.wait_for(
                    cancel.run(work) if cancel is not None else work, timeout=budget
                )
            except TimeoutError:
                result = _result(spec.name, "error", f"browser check timed out after {budget:.0f}s", started)
            except CapabilityUnavailable as exc:
                unavailable = exc.message + (f" ({exc.hint})" if exc.hint else "")
                result = _result(spec.name, "error", unavailable, started)
            results.append(result)
    return results
