"""BrowserManager gating, diagnostics and refusals that must happen before any browser starts.

Everything here runs through a real CoreRuntime and the real ToolExecutor/policy engine; no
Chromium is launched (asserted after each scenario), so these tests belong to the default suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.helpers import Harness, make_repo

import coremain.browser.manager as manager_module
from coremain.browser.manager import BrowserManager
from coremain.browser.origins import TargetError
from coremain.errors import CapabilityUnavailable, PolicyDeniedError, UsageError
from coremain.providers.types import ToolCall
from coremain.runtime.cancel import CancelToken
from coremain.runtime.toolsets import tools_for
from coremain.tools.base import SideEffect, ToolContext, ToolResult
from coremain.tools.executor import ToolExecutor

TOOL_NAMES = [
    "browser_click",
    "browser_console",
    "browser_navigate",
    "browser_screenshot",
    "browser_snapshot",
    "browser_type",
]


@pytest.fixture
def site(tmp_path: Path) -> Path:
    (tmp_path / "outside.html").write_text("<p>outside</p>", encoding="utf-8")
    return make_repo(tmp_path / "site", {"index.html": "<h1>Home</h1>", "README.md": "demo\n"})


def browser(rt: Any) -> BrowserManager:
    mgr = rt.extension("browser")
    assert isinstance(mgr, BrowserManager)
    return mgr


def context(rt: Any, role: str = "implementer") -> ToolContext:
    project = rt.require_project()
    return ToolContext(
        project_id=project.id,
        session_id=None,
        task_id=None,
        attempt_id=None,
        workspace=rt.workspaces.canonical(project),
        fence=None,
        cancel=CancelToken(),
        role=role,
        services=rt.tool_services(project, {}),
    )


async def call(mgr: BrowserManager, ctx: ToolContext, name: str, **arguments: Any) -> ToolResult:
    tools = {t.name: t for t in mgr.tools_for_role(ctx.role)}
    return await ToolExecutor(tools).execute(ToolCall(id=f"call-{name}", name=name, arguments=arguments), ctx)


def assert_not_launched(mgr: BrowserManager) -> None:
    assert mgr._browser is None and mgr._pw is None
    assert not mgr._sessions and not mgr._ephemeral


async def test_tools_are_offered_to_verification_roles_only(harness: Harness, site: Path) -> None:
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        for role in ("implementer", "corrector", "debugger", "reviewer", "researcher"):
            assert sorted(t.name for t in mgr.tools_for_role(role)) == TOOL_NAMES, role
        assert mgr.tools_for_role("planner") == []
        for tool in mgr.tools_for_role("reviewer"):
            assert tool.capability == "browser" and tool.read_only and tool.side_effect == SideEffect.NONE
            assert not any(word in tool.name for word in ("eval", "script", "code"))
        # Read-only roles keep them through the runtime's toolset assembly.
        for role in ("reviewer", "researcher"):
            assert set(TOOL_NAMES) <= {t.name for t in tools_for(role, rt.extra_tools(role))}
        assert not {t.name for t in tools_for("planner", rt.extra_tools("planner"))} & set(TOOL_NAMES)
        schema = {t.name: t.spec() for t in mgr.tools_for_role("implementer")}["browser_click"]
        assert {"ref", "selector", "element"} <= set(schema["parameters"]["properties"])
        assert_not_launched(mgr)


async def test_disabled_browser_offers_nothing_and_says_so(harness: Harness, site: Path) -> None:
    harness.config("[browser]\nenabled = false\n")
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        assert mgr.tools_for_role("implementer") == []
        for live in (False, True):
            status = await mgr.check(live=live)
            assert status["status"] == "disabled" and "browser.enabled = false" in status["detail"]
        results = await mgr.run_checks(
            [{"name": "home", "url": "index.html"}], workspace_path=site, task_id=None
        )
        assert [(r["name"], r["status"]) for r in results] == [("home", "error")]
        assert "disabled" in results[0]["summary"]
        with pytest.raises(CapabilityUnavailable):
            await mgr.screenshot_url("index.html", site / "shot.png")
        assert_not_launched(mgr)


async def test_missing_playwright_is_unavailable_with_an_install_hint(
    harness: Harness, site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(manager_module, "playwright_version", lambda: None)
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        assert mgr.tools_for_role("implementer") == []
        assert all(t.name not in TOOL_NAMES for t in rt.extra_tools("implementer"))
        for live in (False, True):
            status = await mgr.check(live=live)
            assert status["status"] == "unavailable"
            assert "playwright install chromium" in status["detail"]
        results = await mgr.run_checks(
            [{"name": "home", "url": "index.html"}], workspace_path=site, task_id=None
        )
        assert results[0]["status"] == "error" and "not installed" in results[0]["summary"]
        assert_not_launched(mgr)


async def test_missing_executable_and_invalid_origins_are_diagnosed(harness: Harness, site: Path) -> None:
    harness.config(
        '[browser]\nexecutable_path = "/nonexistent/chromium"\n'
        'allowed_origins = ["ftp://files.example", "http://localhost:*"]\n'
    )
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        status = await mgr.check(live=True)
        assert status["status"] == "unavailable" and "/nonexistent/chromium" in status["detail"]
        assert status["invalid_origins"] and "ftp://files.example" in status["invalid_origins"][0]
        assert_not_launched(mgr)


async def test_check_without_live_reports_available_without_launching(harness: Harness, site: Path) -> None:
    harness.config('[browser]\nallowed_origins = ["http://localhost:*", "gopher://x"]\n')
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        status = await mgr.check()
        assert status["status"] == "available"
        assert status["playwright"] and status["allowed_origins"] == ["http://localhost:*", "gopher://x"]
        assert "gopher://x" in status["detail"]  # invalid patterns are surfaced, never silently used
        assert_not_launched(mgr)


async def test_tool_refusals_happen_before_any_browser_starts(harness: Harness, site: Path) -> None:
    harness.config('[permissions]\nprofile = "autonomous"\n')
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        ctx = context(rt)
        escape = await call(mgr, ctx, "browser_navigate", url="../outside.html")
        assert (escape.status, escape.error_class) == ("denied", "path_violation")
        absolute = await call(mgr, ctx, "browser_navigate", url=(site.parent / "outside.html").as_uri())
        assert absolute.status == "denied" and "outside the workspace" in absolute.content
        remote = await call(mgr, ctx, "browser_navigate", url="https://example.com/login")
        assert remote.status == "denied" and "not in browser.allowed_origins" in remote.content
        secret = site / ".env"
        secret.write_text("TOKEN=abc\n", encoding="utf-8")
        leaked = await call(mgr, ctx, "browser_navigate", url=".env")
        assert leaked.status == "denied" and "sensitive" in leaked.content
        bad = await call(mgr, ctx, "browser_navigate", url="javascript:alert(1)")
        assert bad.error_class == "invalid_target"
        hint = await call(mgr, ctx, "browser_navigate", url="localhost:3000")
        assert "did you mean http://localhost:3000?" in hint.content
        for name in ("browser_snapshot", "browser_console", "browser_screenshot"):
            result = await call(mgr, ctx, name)
            assert (result.ok, result.error_class) == (False, "no_page"), name
        click = await call(mgr, ctx, "browser_click", ref="e3")
        assert click.error_class == "no_page"
        both = await call(mgr, ctx, "browser_click", ref="e3", selector="#x")
        neither = await call(mgr, ctx, "browser_type", text="hi")
        assert both.error_class == neither.error_class == "invalid_arguments"
        assert "exactly one of 'ref' or 'selector'" in both.content
        assert mgr.current_target(ctx) == ""
        assert_not_launched(mgr)


async def test_policy_engine_governs_browser_tools(harness: Harness, site: Path) -> None:
    harness.config(
        '[browser]\nallowed_origins = ["*"]\n\n'
        '[[permissions.rules]]\ncapability = "browser"\npattern = "http://localhost:4000"\ndecision = "deny"\n'
    )
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        ctx = context(rt)
        # Standard profile: a non-loopback origin needs approval, which nobody can give here.
        remote = await call(mgr, ctx, "browser_navigate", url="https://example.com/login")
        assert (remote.status, remote.error_class) == ("denied", "approval_rejected")
        approvals = rt.db.query("SELECT capability, request_json FROM approvals")
        assert [r["capability"] for r in approvals] == ["browser"]
        assert '"target": "https://example.com"' in approvals[0]["request_json"]
        ruled = await call(mgr, ctx, "browser_navigate", url="http://localhost:4000/admin")
        assert ruled.status == "denied" and "rule" in ruled.content
        assert_not_launched(mgr)


async def test_offline_and_read_only_profiles_refuse_navigation(harness: Harness, site: Path) -> None:
    harness.config('[browser]\nallowed_origins = ["*"]\n\n[permissions.network]\noffline = true\n')
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        result = await call(mgr, context(rt), "browser_navigate", url="https://example.com/")
        assert result.status == "denied" and "offline" in result.content
        with pytest.raises(PolicyDeniedError, match="offline"):
            await mgr.screenshot_url("https://example.com/", site / "shot.png")
        checks = await mgr.run_checks(
            [{"name": "remote", "url": "https://example.com/"}], workspace_path=site, task_id=None
        )
        assert checks[0]["status"] == "error" and "offline" in checks[0]["summary"]
        assert not (site / "shot.png").exists()
        assert_not_launched(mgr)
    harness.config('[permissions]\nprofile = "read-only"\n')
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        result = await call(mgr, context(rt, "reviewer"), "browser_navigate", url="index.html")
        assert result.status == "denied" and "read-only" in result.content
        assert_not_launched(mgr)


async def test_cli_screenshot_refusals(harness: Harness, site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness.config('[[permissions.rules]]\ncapability = "browser"\npattern = "file://*"\ndecision = "deny"\n')
    monkeypatch.chdir(site)
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        with pytest.raises(PolicyDeniedError, match="denied"):
            await mgr.screenshot_url("index.html", Path("shot.png"))
        with pytest.raises(PolicyDeniedError, match=r"not in browser\.allowed_origins"):
            await mgr.screenshot_url(
                "https://example.com/", Path("shot.png")
            )  # "ask" is consent, origins are not
        with pytest.raises(TargetError):
            await mgr.screenshot_url("missing.html", Path("shot.png"))
        with pytest.raises(UsageError):
            await mgr.screenshot_url("http://localhost:3000/", Path("shot.png"), width=5)
        assert not (site / "shot.png").exists()
        assert_not_launched(mgr)


async def test_invalid_checks_fail_closed_without_launching(harness: Harness, site: Path) -> None:
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        checks: list[Any] = [
            {"name": "typo", "url": "index.html", "expect_txt": "Home"},
            {"url": "index.html"},
            "index.html",
            {"name": "escape", "url": "../outside.html"},
            {"name": "remote", "url": "https://example.com/"},
            {"name": "remote", "url": "index.html"},
            {"name": "scheme", "url": "javascript:alert(1)"},
        ]
        results = await mgr.run_checks(checks, workspace_path=site, task_id=None)
        assert [r["status"] for r in results] == ["error"] * len(checks)
        by_name = {r["name"]: r["summary"] for r in results}
        assert "expect_txt" in by_name["typo"]
        assert "check-2" in by_name and "check-3" in by_name
        assert "outside the workspace" in by_name["escape"]
        assert "policy ask" in results[4]["summary"]
        assert results[5]["summary"] == "duplicate browser check name"
        assert "unsupported URL scheme" in by_name["scheme"]
        assert_not_launched(mgr)


async def test_lifecycle_calls_are_safe_without_sessions(harness: Harness, site: Path) -> None:
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        assert await mgr.close_task("task-that-never-browsed") is False
        await mgr.aclose()
        await mgr.aclose()
        assert_not_launched(mgr)
