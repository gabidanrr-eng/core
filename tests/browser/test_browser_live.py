"""Live browser tests: real headless Chromium through the real runtime, executor and policy.

Run with ``uv run pytest -m browser tests/browser``. Every scenario also asserts that nothing
reached the off-limits server, so an enforcement gap shows up as a failure rather than a pass.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from coremain.browser.manager import BrowserManager
from coremain.errors import PolicyDeniedError
from coremain.providers.types import ToolCall
from coremain.runtime.cancel import CancelToken
from coremain.tools.base import ToolContext, ToolResult
from coremain.tools.executor import ToolExecutor
from tests.browser.conftest import Web
from tests.helpers import Harness, make_repo

pytestmark = pytest.mark.browser
PNG = b"\x89PNG\r\n\x1a\n"


def demo_page(outside_url: str) -> str:
    return (
        "<!doctype html><html><head><title>Demo</title></head><body>"
        "<h1>Hello demo</h1><p id='status'>idle</p>"
        "<button onclick=\"document.getElementById('status').textContent = 'clicked!'\">Press me</button>"
        "<form onsubmit=\"event.preventDefault(); document.getElementById('status').textContent = "
        "'hi ' + document.getElementById('name').value\"><input id='name' aria-label='Your name'></form>"
        "<button onclick=\"console.error('boom from page'); undefinedFunction()\">Break</button>"
        "<button onclick=\"alert('are you sure?')\">Warn</button>"
        f"<iframe title='outside' src='{outside_url}'></iframe><iframe title='secret' src='.env'></iframe>"
        "</body></html>"
    )


@pytest.fixture
def site(tmp_path: Path) -> Path:
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("TOP-SECRET-OUTSIDE-WORKSPACE", encoding="utf-8")
    repo = make_repo(tmp_path / "site", {"index.html": demo_page(outside.as_uri()), "README.md": "demo\n"})
    (repo / ".env").write_text("TOKEN=workspace-secret-value\n", encoding="utf-8")
    return repo


def browser(rt: Any) -> BrowserManager:
    mgr = rt.extension("browser")
    assert isinstance(mgr, BrowserManager)
    return mgr


def context(
    rt: Any, *, task_id: str | None = None, workspace: Any = None, role: str = "implementer"
) -> ToolContext:
    project = rt.require_project()
    return ToolContext(
        project_id=project.id,
        session_id=None,
        task_id=task_id,
        attempt_id=None,
        workspace=workspace or rt.workspaces.canonical(project),
        fence=None,
        cancel=CancelToken(),
        role=role,
        services=rt.tool_services(project, {}),
    )


def new_task(rt: Any, title: str) -> str:
    project = rt.require_project()
    return str(rt.tasks.create(project_id=project.id, title=title, description=title, kind="feature").id)


async def call(mgr: BrowserManager, ctx: ToolContext, name: str, **arguments: Any) -> ToolResult:
    tools = {t.name: t for t in mgr.tools_for_role(ctx.role)}
    return await ToolExecutor(tools).execute(ToolCall(id=f"call-{name}", name=name, arguments=arguments), ctx)


def ref(content: str, role: str, name: str) -> str:
    match = re.search(rf'{role} "{re.escape(name)}"[^\n]*?\[ref=([a-z0-9]+)\]', content)
    assert match, f"no {role} {name!r} in snapshot:\n{content}"
    return match.group(1)


async def test_workspace_page_snapshot_click_type_console_and_screenshot(
    harness: Harness, site: Path
) -> None:
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        ctx = context(rt, role="debugger")
        nav = await call(mgr, ctx, "browser_navigate", url="index.html")
        assert nav.ok, nav.content
        assert "[untrusted web page content" in nav.content and "Hello demo" in nav.content
        # Neither the file outside the workspace nor the workspace .env may reach the model.
        assert "TOP-SECRET" not in nav.content and "workspace-secret-value" not in nav.content
        assert "file is outside the workspace" in nav.content and "sensitive file" in nav.content

        clicked = await call(
            mgr, ctx, "browser_click", ref=ref(nav.content, "button", "Press me"), element="Press me"
        )
        assert clicked.ok and "clicked!" in clicked.content
        typed = await call(
            mgr, ctx, "browser_type", ref=ref(nav.content, "textbox", "Your name"), text="Ada", submit=True
        )
        assert typed.ok and "hi Ada" in typed.content
        by_selector = await call(mgr, ctx, "browser_click", selector="text=Press me")
        assert by_selector.ok
        stale = await call(mgr, ctx, "browser_click", ref="e999")
        assert stale.error_class == "element_not_found" and "browser_snapshot" in stale.content
        ambiguous = await call(mgr, ctx, "browser_click", selector="button")
        assert ambiguous.error_class == "ambiguous_selector"

        broke = await call(mgr, ctx, "browser_click", ref=ref(nav.content, "button", "Break"))
        assert "console: 2 error(s)" in broke.content
        await call(mgr, ctx, "browser_click", ref=ref(nav.content, "button", "Warn"))
        console = await call(mgr, ctx, "browser_console")
        assert console.ok and console.data is not None and console.data["errors"] == 2
        for expected in (
            "boom from page",
            "undefinedFunction is not defined",
            "alert: are you sure?",
            "[blocked]",
        ):
            assert expected in console.content, expected
        again = await call(mgr, ctx, "browser_console")
        assert "nothing new" in again.content

        shot = await call(mgr, ctx, "browser_screenshot", full_page=True)
        assert shot.ok and shot.artifact_id and shot.images and shot.images[0][0] == "image/png"
        stored = rt.artifacts.get(shot.artifact_id)
        assert (stored.kind, stored.media_type, stored.project_id) == (
            "screenshot",
            "image/png",
            ctx.project_id,
        )
        assert rt.artifacts.read_bytes(shot.artifact_id).startswith(PNG)
        row = rt.db.one("SELECT artifact_id, side_effect FROM tool_calls WHERE tool = 'browser_screenshot'")
        assert row is not None and row["artifact_id"] == shot.artifact_id and row["side_effect"] == "none"

        snapshot = await call(mgr, ctx, "browser_snapshot")
        assert snapshot.ok and "clicked!" in snapshot.content  # the selector click ran after typing
        assert mgr.current_target(ctx) == (site / "index.html").resolve().as_uri()


async def test_http_app_and_origin_enforcement_across_redirects_links_and_sockets(
    harness: Harness, site: Path, web: Web
) -> None:
    harness.config(web.config())
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        ctx = context(rt)
        home = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/")
        assert home.ok and "(HTTP 200)" in home.content and "Welcome home" in home.content
        loaded = await call(mgr, ctx, "browser_click", ref=ref(home.content, "button", "Load data"))
        assert "loaded from api" in loaded.content  # the click waited for the fetch and re-render

        local = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/redirect-local")
        assert (
            local.ok
            and "Final destination" in local.content
            and f"navigated to {web.app}/final" in local.content
        )

        # Redirect hops never reach route handlers; the egress proxy stops them.
        redirected = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/redirect-out")
        assert (redirected.ok, redirected.error_class) == (False, "origin_blocked")
        assert "egress proxy" in redirected.content
        tunnelled = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/redirect-secure-out")
        assert tunnelled.status == "denied" and "blocked" in tunnelled.content

        direct = await call(mgr, ctx, "browser_navigate", url=f"{web.offlimits}/secret")
        assert direct.status == "denied" and "not in browser.allowed_origins" in direct.content

        pixel = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/subresource-out")
        assert pixel.ok and "blocked by the browser origin policy" in pixel.content
        socket_page = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/ws")
        assert socket_page.ok
        console = await call(mgr, ctx, "browser_console")
        assert "websocket failed" in console.content and "egress proxy" in console.content

        home = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/")
        link = await call(mgr, ctx, "browser_click", ref=ref(home.content, "link", "Go external"))
        assert "blocked by the browser origin policy" in link.content

        home = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/")
        tab = await call(mgr, ctx, "browser_click", ref=ref(home.content, "link", "Open tab"))
        assert "a new tab opened" in tab.content and "Final destination" in tab.content

        down = await call(mgr, ctx, "browser_navigate", url=f"{web.dead}/")
        assert down.error_class == "navigation_failed" and "connection refused" in down.content
        events = {r["kind"] for r in rt.db.query("SELECT kind FROM events WHERE kind LIKE 'browser.%'")}
        assert {"browser.launched", "browser.blocked"} <= events
    assert web.hits == []


async def test_offline_mode_blocks_external_requests_even_with_a_wildcard_allowlist(
    harness: Harness, site: Path, web: Web
) -> None:
    harness.config('[browser]\nallowed_origins = ["*"]\n\n[permissions.network]\noffline = true\n')
    (site / "external.html").write_text(
        "<h1>External assets</h1><img alt='cdn' src='https://cdn.example.com/logo.png'>", encoding="utf-8"
    )
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        ctx = context(rt)
        result = await call(mgr, ctx, "browser_navigate", url="external.html")
        assert result.ok and "offline mode" in result.content and "External assets" in result.content
        local = await call(mgr, ctx, "browser_navigate", url=f"{web.app}/")
        assert local.ok and "Welcome home" in local.content
    assert web.hits == []


async def test_declarative_checks_pass_fail_and_error(harness: Harness, site: Path, web: Web) -> None:
    harness.config(web.config())
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        task_id = new_task(rt, "verify the app")
        checks: list[dict[str, Any]] = [
            {
                "name": "home",
                "url": f"{web.app}/",
                "expect_text": "Welcome home",
                "expect_no_console_errors": True,
                "screenshot": True,
            },
            {"name": "late", "url": f"{web.app}/late", "expect_text": "Late content"},
            {"name": "heading", "url": f"{web.app}/", "selector": "h1", "expect_text": "Welcome"},
            {"name": "file", "url": "index.html", "expect_text": "Hello demo"},
            {"name": "missing-text", "url": f"{web.app}/", "expect_text": "Not on the page", "timeout_s": 1},
            {"name": "missing-selector", "url": f"{web.app}/", "selector": "#nope", "timeout_s": 1},
            {"name": "server-error", "url": f"{web.app}/broken"},
            {
                "name": "noisy",
                "url": f"{web.app}/noisy",
                "expect_no_console_errors": True,
                "screenshot": True,
            },
            {"name": "redirect-out", "url": f"{web.app}/redirect-out"},
            {"name": "bad-selector", "url": f"{web.app}/", "selector": "div[[["},
            {"name": "down", "url": f"{web.dead}/"},
            {"name": "outside", "url": "../outside-secret.txt"},
            {"name": "offlimits", "url": f"{web.offlimits}/"},
        ]
        results = await mgr.run_checks(checks, workspace_path=site, task_id=task_id)
        status = {r["name"]: r["status"] for r in results}
        assert status == {
            "home": "pass",
            "late": "pass",
            "heading": "pass",
            "file": "pass",
            "missing-text": "fail",
            "missing-selector": "fail",
            "server-error": "fail",
            "noisy": "fail",
            "redirect-out": "fail",
            "bad-selector": "error",
            "down": "error",
            "outside": "error",
            "offlimits": "error",
        }, results
        by_name = {r["name"]: r for r in results}
        home = by_name["home"]
        assert (
            home["http_status"] == 200
            and home["console_errors"] == []
            and "no console errors" in home["summary"]
        )
        art = rt.artifacts.get(home["artifact_id"])
        assert (art.task_id, art.kind) == (task_id, "screenshot")
        assert rt.artifacts.read_bytes(home["artifact_id"]).startswith(PNG)
        assert "HTTP 500" in by_name["server-error"]["summary"]
        assert "noisy failure" in by_name["noisy"]["console_errors"][0] and by_name["noisy"]["artifact_id"]
        assert "Not on the page" in by_name["missing-text"]["summary"]
        assert "untrusted page text" in by_name["missing-text"]["summary"]
        assert by_name["redirect-out"]["blocked"] and "blocked" in by_name["redirect-out"]["summary"]
        assert "connection refused" in by_name["down"]["summary"]
        assert "invalid selector" in by_name["bad-selector"]["summary"]
        assert all("duration_ms" in r for r in results)
        # Checks use throwaway contexts and leave no browser running afterwards.
        assert mgr._browser is None and not mgr._ephemeral and not mgr._sessions
    assert web.hits == []


async def test_task_sessions_are_isolated_replaced_and_closed(harness: Harness, site: Path, web: Web) -> None:
    harness.config(web.config())
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        first = context(rt, task_id=new_task(rt, "first"))
        second = context(rt, task_id=new_task(rt, "second"))
        home = await call(mgr, first, "browser_navigate", url=f"{web.app}/")
        await call(mgr, first, "browser_click", ref=ref(home.content, "button", "Remember me"))
        again = await call(mgr, first, "browser_navigate", url=f"{web.app}/")
        assert "visitor: first task" in again.content
        other = await call(mgr, second, "browser_navigate", url=f"{web.app}/")
        assert "visitor: none" in other.content  # separate contexts share no storage
        assert len(mgr._sessions) == 2

        # A new attempt in a different workspace never inherits the old session.
        copy = site.parent / "attempt-2"
        shutil.copytree(site, copy)
        moved = context(
            rt,
            task_id=first.task_id,
            workspace=dataclasses.replace(first.workspace, id="ws-attempt-2", path=copy),
        )
        assert (await call(mgr, moved, "browser_snapshot")).error_class == "no_page"
        fresh = await call(mgr, moved, "browser_navigate", url=f"{web.app}/")
        assert "visitor: none" in fresh.content

        assert first.task_id and second.task_id
        assert await mgr.close_task(first.task_id) is True
        assert await mgr.close_task(first.task_id) is False
        assert mgr._browser is not None  # the second task still has a session
        assert (await call(mgr, first, "browser_snapshot")).error_class == "no_page"
        assert await mgr.close_task(second.task_id) is True
        assert mgr._browser is None and mgr._pw is None  # idle runtime holds no browser
    assert web.hits == []


async def test_lost_browser_is_explicit_and_recoverable(harness: Harness, site: Path) -> None:
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        ctx = context(rt)
        assert (await call(mgr, ctx, "browser_navigate", url="index.html")).ok
        await mgr._browser.close()  # simulate the browser process going away
        lost = await call(mgr, ctx, "browser_snapshot")
        assert (lost.ok, lost.error_class) == (False, "browser_session_lost")
        assert "browser_navigate" in lost.content
        events = [
            r["kind"] for r in rt.db.query("SELECT kind FROM events WHERE kind = 'browser.disconnected'")
        ]
        assert events == ["browser.disconnected"]
        recovered = await call(mgr, ctx, "browser_navigate", url="index.html")
        assert recovered.ok and "Hello demo" in recovered.content


async def test_cli_screenshot_and_live_check(
    harness: Harness, site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(site)
    async with harness.runtime(site) as rt:
        mgr = browser(rt)
        status = await mgr.check(live=True)
        assert status["status"] == "ready" and status["browser_version"], status
        info = await mgr.screenshot_url("index.html", Path("out/shot.png"), width=800, height=600)
        saved = site / "out" / "shot.png"
        assert saved.read_bytes().startswith(PNG) and info["path"] == str(saved)
        assert (info["title"], info["width"], info["height"], info["http_status"]) == ("Demo", 800, 600, 200)
        assert info["blocked"]  # the outside-workspace iframe was refused, even for the CLI
        with pytest.raises(PolicyDeniedError, match="sensitive"):
            await mgr.screenshot_url(".env", Path("out/nope.png"))
        assert mgr._browser is None and not mgr._ephemeral
