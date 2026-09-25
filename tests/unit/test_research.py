"""ResearchService and research tools against a real CoreRuntime with a mocked network (no I/O)."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from coremain.errors import CapabilityUnavailable, ExitCode, PolicyDeniedError
from coremain.providers.types import ToolCall
from coremain.research.context7 import DOCS_MAX_CHARS, SNIPPET_SEPARATOR
from coremain.research.errors import ResearchError
from coremain.research.service import ResearchService
from coremain.runtime.app import CoreRuntime
from coremain.runtime.cancel import CancelToken
from coremain.runtime.toolsets import tools_for
from coremain.tools.base import ApprovalDecision, ToolContext
from coremain.tools.executor import ToolExecutor
from coremain.util.clock import FakeClock

if TYPE_CHECKING:
    from tests.helpers import Harness

ROLES = ("planner", "implementer", "corrector", "debugger", "researcher", "reviewer")
C7 = "context7.com"
SEARCH = "/api/v2/libs/search"
CONTEXT = "/api/v2/context"
SEARCH_RESULTS = {
    "results": [
        {
            "id": "/kludex/fastapi-tips",
            "title": "FastAPI Tips",
            "description": "Tips and tricks",
            "state": "finalized",
            "totalSnippets": 44,
            "trustScore": 10,
            "benchmarkScore": 75.9,
            "versions": [],
        },
        {
            "id": "/websites/fastapi_tiangolo",
            "title": "FastAPI",
            "description": "FastAPI framework, high performance",
            "state": "finalized",
            "totalSnippets": 2377,
            "trustScore": 9,
            "benchmarkScore": 87.98,
            "versions": [],
        },
    ],
    "searchFilterApplied": False,
}
CONTEXT_TEXT = (
    "### Dependencies with yield\n\n"
    "Source: https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/\n\n"
    "Run setup before the response and teardown after it.\n\n"
    "```python\nasync def get_db():\n    db = DBSession()\n    try:\n        yield db\n    finally:\n        db.close()\n```\n\n"
    f"{SNIPPET_SEPARATOR}\n\n"
    "### fastapi.Depends\n\nSource: https://fastapi.tiangolo.com/reference/dependencies/\n\nDeclare a dependency."
)
GUIDE_HTML = (
    "<html><head><title>Guide</title><script>tracker()</script></head><body><nav>Menu</nav>"
    "<main><h1>Guide</h1><p>Install with <code>pip install demo</code>.</p>"
    "<p><a href='/3/next.html'>Next page</a></p></main><footer>Footer</footer></body></html>"
)

Handler = Callable[[httpx.Request], httpx.Response]


class FakeWeb:
    """Routes requests by (host, path) and records them; unknown routes answer 404."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Handler] = {}
        self.requests: list[httpx.Request] = []

    def route(self, host: str, path: str, handler: Handler | httpx.Response) -> None:
        self.routes[(host, path)] = handler if callable(handler) else (lambda _req, r=handler: r)

    def context7(self, *, search: Any = SEARCH_RESULTS, context: str = CONTEXT_TEXT) -> FakeWeb:
        self.route(C7, SEARCH, httpx.Response(200, json=search))
        self.route(C7, CONTEXT, httpx.Response(200, text=context, headers={"ratelimit-remaining": "42"}))
        return self

    def html(self, host: str, path: str, body: str, **headers: str) -> None:
        self.route(
            host, path, httpx.Response(200, text=body, headers={"content-type": "text/html", **headers})
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self.routes.get((request.url.host, request.url.path))
        return handler(request) if handler else httpx.Response(404, text="no route")

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def hosts(self) -> set[str]:
        return {r.url.host for r in self.requests}


@contextlib.asynccontextmanager
async def research(
    harness: Harness, web: FakeWeb, project: Path | None = None, **kw: Any
) -> AsyncIterator[tuple[CoreRuntime, ResearchService]]:
    async with harness.runtime(project, **kw) as rt:
        service = ResearchService(rt, transport=httpx.MockTransport(web))
        service.retry_backoff_s = 0.0
        try:
            yield rt, service
        finally:
            await service.aclose()


def events(rt: CoreRuntime, kind: str) -> list[dict[str, Any]]:
    rows = rt.db.query("SELECT data_json FROM events WHERE kind = ? ORDER BY seq", (kind,))
    return [json.loads(r["data_json"]) for r in rows]


# ================================================================================ docs
async def test_docs_resolves_the_best_library_and_returns_bounded_context(harness: Harness) -> None:
    web = FakeWeb().context7()
    async with research(harness, web) as (rt, svc):
        result = await svc.docs("FastAPI", "dependencies  with\nyield")

        assert result["library_id"] == "/websites/fastapi_tiangolo"
        assert result["source"] == "context7"
        assert result["cached"] is False
        assert "yield db" in result["text"]
        assert result["url"] == "https://context7.com/websites/fastapi_tiangolo"
        assert result["sources"] == [
            "https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/",
            "https://fastapi.tiangolo.com/reference/dependencies/",
        ]
        # The exact name match beats the candidate Context7 ranked first, and the choice is explained.
        assert [c["id"] for c in result["candidates"]] == [
            "/websites/fastapi_tiangolo",
            "/kludex/fastapi-tips",
        ]
        assert [c["selected"] for c in result["candidates"]] == [True, False]
        assert "exact name match" in result["selection_reason"]
        assert result["rate_limit"] == {"remaining": 42}

        search, context = web.requests
        assert (search.url.path, search.url.params["libraryName"], search.url.params["query"]) == (
            SEARCH,
            "FastAPI",
            "dependencies with yield",
        )
        assert context.url.params["libraryId"] == "/websites/fastapi_tiangolo"
        assert "authorization" not in context.headers
        assert context.headers["user-agent"].startswith("core-main/")
        [event] = events(rt, "research.docs")
        assert (
            event["ok"] is True
            and event["library_id"] == "/websites/fastapi_tiangolo"
            and event["via"] == "user"
        )


async def test_direct_library_id_skips_search_and_follows_library_redirects(harness: Harness) -> None:
    web = FakeWeb()

    def context(request: httpx.Request) -> httpx.Response:
        if request.url.params["libraryId"] == "/fastapi/fastapi":
            return httpx.Response(
                301,
                json={
                    "error": "library_redirected",
                    "message": "Library /fastapi/fastapi has been redirected",
                    "redirectUrl": "/websites/fastapi_tiangolo",
                },
            )
        return httpx.Response(
            200,
            json={
                "codeSnippets": [
                    {
                        "codeTitle": "Path Parameters",
                        "codeId": "https://fastapi.tiangolo.com/tutorial/path-params/",
                        "codeList": [{"language": "python", "code": "@app.get('/items/{item_id}')"}],
                    }
                ],
                "infoSnippets": [],
            },
        )

    web.route(C7, CONTEXT, context)
    async with research(harness, web) as (_rt, svc):
        result = await svc.docs("/fastapi/fastapi", "path parameters")
    assert web.paths() == [CONTEXT, CONTEXT]
    assert result["library_id"] == "/websites/fastapi_tiangolo"
    assert result["redirected_from"] == "/fastapi/fastapi"
    assert result["candidates"] == [] and result["selection_reason"] == "explicit Context7 library id"
    assert "### Path Parameters" in result["text"] and "```python\n@app.get" in result["text"]


async def test_docs_text_is_bounded_at_snippet_boundaries(harness: Harness) -> None:
    big = f"\n\n{SNIPPET_SEPARATOR}\n\n".join(
        f"### Snippet {i}\n\nSource: https://example.com/{i}\n\n" + "x" * 3000 for i in range(20)
    )
    web = FakeWeb().context7(context=big)
    async with research(harness, web) as (_rt, svc):
        result = await svc.docs("/org/lib", "anything")
    assert len(result["text"]) <= DOCS_MAX_CHARS
    assert result["truncated"] is True
    assert result["snippets"]["total"] == 20
    included = result["snippets"]["included"]
    assert 0 < included < 20
    assert result["text"].count("x" * 3000) == included
    assert "more snippet(s) omitted" in result["text"]


async def test_docs_cache_hits_resolution_reuse_and_ttl_expiry(harness: Harness) -> None:
    harness.config("[research]\ncache_ttl_s = 60\n")
    clock = FakeClock()
    web = FakeWeb().context7()
    async with research(harness, web, clock=clock) as (rt, svc):
        first = await svc.docs("fastapi", "yield")
        assert web.paths() == [SEARCH, CONTEXT]

        second = await svc.docs("fastapi", "yield")
        assert second["cached"] is True
        assert second["text"] == first["text"] and second["library_id"] == first["library_id"]
        assert len(web.requests) == 2
        assert rt.db.scalar("SELECT hits FROM cache_entries WHERE namespace = 'research.docs'") == 1

        # A different question reuses the cached library resolution.
        await svc.docs("fastapi", "routing")
        assert web.paths()[2:] == [CONTEXT]

        clock.advance(61)
        third = await svc.docs("fastapi", "yield")
        assert third["cached"] is False
        assert web.paths()[3:] == [SEARCH, CONTEXT]
        assert svc.describe()["cache"]["research.docs"]["entries"] == 1  # expired rows were purged


async def test_zero_ttl_disables_caching(harness: Harness) -> None:
    harness.config("[research]\ncache_ttl_s = 0\n")
    web = FakeWeb().context7()
    async with research(harness, web) as (rt, svc):
        await svc.docs("/org/lib", "q")
        again = await svc.docs("/org/lib", "q")
        assert again["cached"] is False
        assert web.paths() == [CONTEXT, CONTEXT]
        assert rt.db.scalar("SELECT COUNT(*) FROM cache_entries") == 0


async def test_corrupted_cache_entry_degrades_to_a_miss(harness: Harness) -> None:
    web = FakeWeb().context7()
    async with research(harness, web) as (rt, svc):
        await svc.docs("/org/lib", "q")
        rt.db.execute(
            "UPDATE cache_entries SET value = ? WHERE namespace = 'research.docs'", (b"\x00{not json",)
        )
        again = await svc.docs("/org/lib", "q")
        assert again["cached"] is False and "yield db" in again["text"]
        assert web.paths() == [CONTEXT, CONTEXT]
        assert (await svc.docs("/org/lib", "q"))["cached"] is True


async def test_disabled_docs_provider_is_an_explicit_unavailable_capability(harness: Harness) -> None:
    harness.config('[research]\ndocs_provider = "none"\n')
    web = FakeWeb().context7()
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(CapabilityUnavailable, match="disabled") as exc:
            await svc.docs("fastapi", "q")
        assert exc.value.exit_code == ExitCode.BLOCKED
        assert [t.name for t in svc.tools_for_role("researcher")] == ["web_fetch"]
        assert svc.describe()["status"] == "docs disabled"
    assert web.requests == []


async def test_api_key_is_sent_as_bearer_and_never_leaks(harness: Harness) -> None:
    secret = "ctx7sk-3f9a1c2e8b7d4e6fa1b2c3d4e5f60718"
    harness.env["CONTEXT7_API_KEY"] = secret
    harness.config('[research]\ncontext7_api_key = "env:CONTEXT7_API_KEY"\n')
    web = FakeWeb()
    seen: list[str] = []

    def search(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(401, json={"error": "invalid_api_key", "message": f"Invalid API key {secret}"})

    web.route(C7, SEARCH, search)
    async with research(harness, web) as (rt, svc):
        with pytest.raises(ResearchError) as exc:
            await svc.docs("fastapi", "q")
        err = exc.value
        assert seen == [f"Bearer {secret}"]
        assert err.error_class == "auth_failed" and err.status == 401 and err.exit_code == ExitCode.BLOCKED
        assert "invalid_api_key" in err.message
        assert secret not in err.message and secret not in (err.hint or "")
        assert "research.context7_api_key" in (err.hint or "")
        [event] = events(rt, "research.docs")
        assert event["ok"] is False and event["error_class"] == "auth_failed"
        assert secret not in json.dumps(event)
        assert "env: present" in svc.describe()["credential"]


async def test_unresolvable_api_key_reference_fails_closed(harness: Harness) -> None:
    harness.config('[research]\ncontext7_api_key = "env:MISSING_CONTEXT7_KEY"\n')
    web = FakeWeb().context7()
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(ResearchError) as exc:
            await svc.docs("fastapi", "q")
    assert exc.value.error_class == "credential_missing"
    assert "MISSING_CONTEXT7_KEY" in exc.value.message
    assert web.requests == []


@pytest.mark.parametrize(
    ("status", "body", "headers", "error_class", "exit_code", "attempts"),
    [
        (
            429,
            {"error": "rate_limited", "message": "Too many"},
            {"retry-after": "17"},
            "rate_limited",
            ExitCode.BLOCKED,
            1,
        ),
        (401, {"error": "invalid_api_key", "message": "bad key"}, {}, "auth_failed", ExitCode.BLOCKED, 1),
        (403, {"error": "forbidden", "message": "private"}, {}, "auth_failed", ExitCode.BLOCKED, 1),
        (404, {"error": "library_not_found", "message": "nope"}, {}, "not_found", ExitCode.NOT_FOUND, 1),
        (400, {"error": "validation_error", "message": "bad"}, {}, "invalid_request", ExitCode.USAGE, 1),
        (202, {"error": "not_finalized", "message": "wait"}, {}, "not_ready", ExitCode.BLOCKED, 1),
        (500, {"error": "internal", "message": "boom"}, {}, "upstream_error", ExitCode.BLOCKED, 2),
        (503, {"error": "search_failed", "message": "later"}, {}, "upstream_error", ExitCode.BLOCKED, 2),
    ],
)
async def test_context7_http_errors_map_to_typed_failures(
    harness: Harness,
    status: int,
    body: dict[str, str],
    headers: dict[str, str],
    error_class: str,
    exit_code: ExitCode,
    attempts: int,
) -> None:
    web = FakeWeb()
    web.route(C7, SEARCH, httpx.Response(status, json=body, headers=headers))
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(ResearchError) as exc:
            await svc.docs("fastapi", "q")
    err = exc.value
    assert (err.error_class, err.status, err.exit_code) == (error_class, status, exit_code)
    assert len(web.requests) == attempts
    assert f"HTTP {status}" in err.message
    if status == 429:
        assert err.retry_after_s == 17 and "retry after 17s" in err.message
        assert "research.context7_api_key" in (err.hint or "")


async def test_malformed_search_response_is_classified(harness: Harness) -> None:
    web = FakeWeb()
    web.route(C7, SEARCH, httpx.Response(200, text="<html>maintenance</html>"))
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(ResearchError) as exc:
            await svc.docs("fastapi", "q")
    assert exc.value.error_class == "invalid_response"


async def test_network_failures_are_retried_once_then_reported(harness: Harness) -> None:
    calls = {"connect": 0, "read": 0}

    def refuse(request: httpx.Request) -> httpx.Response:
        calls["connect"] += 1
        raise httpx.ConnectError("connection refused", request=request)

    def stall(request: httpx.Request) -> httpx.Response:
        calls["read"] += 1
        raise httpx.ReadTimeout("read timed out", request=request)

    web = FakeWeb()
    web.route(C7, SEARCH, refuse)
    web.route("docs.python.org", "/slow", stall)
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(ResearchError) as exc:
            await svc.docs("fastapi", "q")
        assert exc.value.error_class == "network_error" and "context7.com" in exc.value.message
        with pytest.raises(ResearchError) as slow:
            await svc.fetch("https://docs.python.org/slow")
        assert slow.value.error_class == "timeout"
    assert calls == {"connect": 2, "read": 1}


# =============================================================================== fetch
async def test_fetch_extracts_readable_text_title_and_links(harness: Harness) -> None:
    web = FakeWeb()
    web.html("docs.python.org", "/3/guide.html", GUIDE_HTML)
    async with research(harness, web) as (rt, svc):
        page = await svc.fetch("https://docs.python.org/3/guide.html#install")
        assert page["title"] == "Guide"
        assert page["text"] == "# Guide\n\nInstall with `pip install demo`.\n\nNext page"
        assert page["links"] == [{"text": "Next page", "url": "https://docs.python.org/3/next.html"}]
        assert (page["status"], page["content_type"], page["cached"], page["truncated"]) == (
            200,
            "text/html",
            False,
            False,
        )
        assert page["url"] == page["final_url"] == "https://docs.python.org/3/guide.html"
        assert web.requests[0].headers["accept"].startswith("text/html")
        [event] = events(rt, "research.fetch")
        assert event["ok"] is True and event["status"] == 200


async def test_fetch_streams_and_stops_at_max_fetch_bytes(harness: Harness) -> None:
    harness.config("[research]\nmax_fetch_bytes = 4000\n")
    produced: list[int] = []

    async def body() -> AsyncIterator[bytes]:
        for i in range(200):
            produced.append(i)
            yield f"<p>chunk {i:04d} {'y' * 980}</p>\n".encode()

    web = FakeWeb()
    web.route(
        "docs.python.org",
        "/big.html",
        lambda _req: httpx.Response(200, headers={"content-type": "text/html"}, content=body()),
    )
    async with research(harness, web) as (_rt, svc):
        page = await svc.fetch("https://docs.python.org/big.html")
    assert page["truncated"] is True and page["bytes"] == 4000
    assert len(produced) <= 6
    assert "chunk 0000" in page["text"] and "chunk 0199" not in page["text"]


async def test_fetch_returns_plain_text_and_rejects_binary(harness: Harness) -> None:
    web = FakeWeb()
    web.route(
        "raw.githubusercontent.com", "/o/r/main/README.md", httpx.Response(200, text="# Readme\n\nplain")
    )
    web.route(
        "docs.python.org",
        "/logo.png",
        httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG\r\n\x1a\n\x00\x00"),
    )
    async with research(harness, web) as (_rt, svc):
        readme = await svc.fetch("https://raw.githubusercontent.com/o/r/main/README.md")
        assert readme["text"] == "# Readme\n\nplain" and readme["title"] is None
        with pytest.raises(ResearchError) as exc:
            await svc.fetch("https://docs.python.org/logo.png")
    assert exc.value.error_class == "unsupported_content"


@pytest.mark.parametrize(
    ("status", "headers", "error_class", "attempts"),
    [
        (404, {}, "not_found", 1),
        (403, {}, "auth_failed", 1),
        (429, {"retry-after": "5"}, "rate_limited", 1),
        (502, {}, "upstream_error", 2),
        (418, {}, "http_error", 1),
    ],
)
async def test_fetch_http_errors_map_to_typed_failures(
    harness: Harness, status: int, headers: dict[str, str], error_class: str, attempts: int
) -> None:
    web = FakeWeb()
    web.route("docs.python.org", "/x", httpx.Response(status, text="error page", headers=headers))
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(ResearchError) as exc:
            await svc.fetch("https://docs.python.org/x")
    assert exc.value.error_class == error_class and exc.value.status == status
    assert len(web.requests) == attempts
    if status == 429:
        assert exc.value.retry_after_s == 5


async def test_fetch_rechecks_every_redirect_hop(harness: Harness) -> None:
    harness.config('[permissions.network]\ndeny_domains = ["evil.example"]\n')
    web = FakeWeb()
    web.route(
        "docs.python.org", "/moved", httpx.Response(302, headers={"location": "https://evil.example/steal"})
    )
    web.route(
        "docs.python.org", "/meta", httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})
    )
    web.route("docs.python.org", "/file", httpx.Response(302, headers={"location": "file:///etc/passwd"}))
    web.route("docs.python.org", "/loop", httpx.Response(302, headers={"location": "/loop"}))
    web.route("docs.python.org", "/old", httpx.Response(301, headers={"location": "/new"}))
    web.html("docs.python.org", "/new", "<p>" + "moved content " * 3 + "</p>")
    web.route(
        "docs.python.org",
        "/to-local",
        httpx.Response(302, headers={"location": "http://localhost:2375/containers/json"}),
    )
    web.route(
        "docs.python.org", "/to-lan", httpx.Response(307, headers={"location": "http://10.0.0.5/admin"})
    )
    web.route("localhost", "/a", httpx.Response(302, headers={"location": "/b"}))
    web.html("localhost", "/b", "<p>local to local is fine</p>")
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(PolicyDeniedError, match=re.escape("evil.example")):
            await svc.fetch("https://docs.python.org/moved")
        with pytest.raises(PolicyDeniedError, match="metadata"):
            await svc.fetch("https://docs.python.org/meta")
        for path in ("/to-local", "/to-lan"):
            with pytest.raises(PolicyDeniedError, match="local/private"):
                await svc.fetch(f"https://docs.python.org{path}")
        local = await svc.fetch("http://localhost:8000/a")
        assert local["final_url"] == "http://localhost:8000/b" and "local to local" in local["text"]
        with pytest.raises(ResearchError) as bad_scheme:
            await svc.fetch("https://docs.python.org/file")
        assert bad_scheme.value.error_class == "invalid_url"
        with pytest.raises(ResearchError) as loop:
            await svc.fetch("https://docs.python.org/loop")
        assert loop.value.error_class == "redirect_limit"
        page = await svc.fetch("https://docs.python.org/old")
    assert page["final_url"] == "https://docs.python.org/new"
    assert page["redirects"] == ["https://docs.python.org/new"]
    assert "moved content" in page["text"]
    assert web.hosts() == {"docs.python.org", "localhost"}
    assert [r.url.path for r in web.requests if r.url.host == "localhost"] == ["/a", "/b"]


async def test_fetch_policy_deny_and_user_initiated_ask(harness: Harness) -> None:
    harness.config(
        '[permissions.network]\ndeny_domains = ["evil.example", "context7.com"]\n\n'
        '[[permissions.rules]]\ncapability = "net.http"\npattern = "pypi.org"\ndecision = "deny"\n'
    )
    web = FakeWeb().context7()
    web.html("example.net", "/page", "<p>not on the allow list, but the user asked for it</p>")
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(PolicyDeniedError, match=re.escape("evil.example")):
            await svc.fetch("https://evil.example/page")
        with pytest.raises(PolicyDeniedError, match="rule"):
            await svc.fetch("https://pypi.org/project/httpx/")
        with pytest.raises(PolicyDeniedError, match=re.escape("context7.com")):
            await svc.docs("fastapi", "q")
        assert web.requests == []
        page = await svc.fetch("https://example.net/page")
    assert "user asked for it" in page["text"]


async def test_fetch_cache_skips_local_targets_and_no_store(harness: Harness) -> None:
    web = FakeWeb()
    web.html("docs.python.org", "/a.html", "<p>cached page</p>")
    web.html("docs.python.org", "/private.html", "<p>no store</p>", **{"cache-control": "no-store"})
    web.html("localhost", "/health", "<p>dev server</p>")
    async with research(harness, web) as (_rt, svc):
        for url in (
            "https://docs.python.org/a.html",
            "https://docs.python.org/private.html",
            "http://localhost:8000/health",
        ):
            await svc.fetch(url)
            second = await svc.fetch(url)
            assert second["cached"] is (url.endswith("a.html")), url
    assert web.paths() == ["/a.html", "/private.html", "/private.html", "/health", "/health"]


async def test_offline_mode_refuses_network_and_serves_only_permitted_cache(harness: Harness) -> None:
    web = FakeWeb().context7()
    web.html("docs.python.org", "/3/guide.html", GUIDE_HTML)
    web.html("localhost", "/health", "<p>local dev server</p>")
    async with research(harness, web) as (_rt, svc):
        await svc.docs("/org/lib", "q")
        await svc.fetch("https://docs.python.org/3/guide.html")
    online_requests = len(web.requests)

    harness.config("[permissions.network]\noffline = true\n")
    async with research(harness, web) as (_rt, svc):
        assert all(svc.tools_for_role(role) == [] for role in ROLES)
        assert svc.describe()["status"] == "offline"
        docs = await svc.docs("/org/lib", "q")
        assert docs["cached"] is True and docs["offline"] is True and "yield db" in docs["text"]
        page = await svc.fetch("https://docs.python.org/3/guide.html")
        assert page["offline"] is True and page["title"] == "Guide"
        with pytest.raises(CapabilityUnavailable, match="offline") as exc:
            await svc.docs("/org/lib", "a question that was never cached")
        assert exc.value.exit_code == ExitCode.BLOCKED
        with pytest.raises(CapabilityUnavailable, match="offline"):
            await svc.fetch("https://docs.python.org/3/other.html")
        assert len(web.requests) == online_requests
        # Loopback targets stay reachable offline (the policy invariant allows them).
        local = await svc.fetch("http://localhost:8000/health")
        assert "local dev server" in local["text"]

    harness.config('[permissions.network]\noffline = true\ndeny_domains = ["docs.python.org"]\n')
    async with research(harness, web) as (_rt, svc):
        with pytest.raises(PolicyDeniedError):
            await svc.fetch("https://docs.python.org/3/guide.html")


# =============================================================================== tools
async def test_tools_for_role_gating_and_runtime_wiring(harness: Harness) -> None:
    web = FakeWeb()
    async with research(harness, web) as (rt, svc):
        names = {role: sorted(t.name for t in svc.tools_for_role(role)) for role in ROLES}
        assert names == {
            "planner": ["docs_lookup", "web_fetch"],
            "implementer": ["docs_lookup", "web_fetch"],
            "corrector": ["docs_lookup", "web_fetch"],
            "debugger": ["docs_lookup", "web_fetch"],
            "researcher": ["docs_lookup", "web_fetch"],
            "reviewer": ["docs_lookup"],
        }
        assert svc.tools_for_role("summarizer") == []
        docs_tool, fetch_tool = svc.tools_for_role("planner")
        assert (docs_tool.capability, fetch_tool.capability) == ("net.docs", "net.http")
        assert docs_tool.read_only and fetch_tool.read_only
        assert docs_tool.target(docs_tool.Input(library="x", query="y")) == "https://context7.com/api"

        # The runtime builds the extension lazily and hands its tools to role toolsets.
        extension = rt.extension("research")
        assert isinstance(extension, ResearchService)
        reviewer = {t.name for t in tools_for("reviewer", rt.extra_tools("reviewer"))}
        researcher = {t.name for t in tools_for("researcher", rt.extra_tools("researcher"))}
        assert "docs_lookup" in reviewer and "web_fetch" not in reviewer
        assert {"docs_lookup", "web_fetch"} <= researcher


def _tool_context(rt: CoreRuntime) -> ToolContext:
    project = rt.require_project()
    return ToolContext(
        project_id=project.id,
        session_id=None,
        task_id=None,
        attempt_id=None,
        workspace=rt.workspaces.canonical(project),
        fence=None,
        cancel=CancelToken(),
        role="researcher",
        services=rt.tool_services(project, {}),
    )


async def test_tools_run_through_the_real_executor(harness: Harness, calc_repo: Path) -> None:
    harness.config(
        '[[permissions.rules]]\ncapability = "net.http"\npattern = "pypi.org"\ndecision = "deny"\n'
    )
    web = FakeWeb().context7()
    web.html("docs.python.org", "/3/guide.html", GUIDE_HTML)
    web.route(
        "docs.python.org", "/r", httpx.Response(302, headers={"location": "https://example.net/landing"})
    )
    web.html("example.net", "/page", "<p>approved external page</p>")
    async with research(harness, web, calc_repo) as (rt, svc):
        ctx = _tool_context(rt)
        executor = ToolExecutor({t.name: t for t in svc.tools_for_role("researcher")})

        docs = await executor.execute(
            ToolCall("c1", "docs_lookup", {"library": "fastapi", "query": "yield"}), ctx
        )
        assert docs.ok, docs.content
        assert "library: /websites/fastapi_tiangolo" in docs.content
        assert "other candidates: /kludex/fastapi-tips (FastAPI Tips)" in docs.content
        nonce = re.search(r"<<<untrusted-([0-9a-f]{8}) ", docs.content)
        assert nonce is not None and docs.content.rstrip().endswith(f"<<<end-untrusted-{nonce.group(1)}>>>")
        assert (
            "yield db" in docs.content
            and docs.data
            and docs.data["library_id"] == "/websites/fastapi_tiangolo"
        )

        page = await executor.execute(
            ToolCall("c2", "web_fetch", {"url": "https://docs.python.org/3/guide.html"}), ctx
        )
        assert page.ok, page.content
        assert "title: Guide" in page.content and "Install with `pip install demo`." in page.content
        assert "- Next page: https://docs.python.org/3/next.html" in page.content

        bad = await executor.execute(ToolCall("c3", "web_fetch", {"url": "file:///etc/passwd"}), ctx)
        assert not bad.ok and bad.error_class == "invalid_arguments"

        # Not on the allow list: policy asks, nobody can approve non-interactively, nothing is fetched.
        ask = await executor.execute(ToolCall("c4", "web_fetch", {"url": "https://example.net/page"}), ctx)
        assert not ask.ok and ask.error_class == "approval_rejected"

        # A redirect from an allowed host to an unapproved one is refused mid-request.
        hop = await executor.execute(ToolCall("c5", "web_fetch", {"url": "https://docs.python.org/r"}), ctx)
        assert not hop.ok and hop.status == "denied" and "needs approval" in hop.content

        denied = await executor.execute(
            ToolCall("c6", "web_fetch", {"url": "https://pypi.org/project/x/"}), ctx
        )
        assert not denied.ok and denied.error_class == "policy_denied"
        assert "example.net" not in web.hosts() and "pypi.org" not in web.hosts()

        rows = rt.db.query("SELECT tool, capability, status FROM tool_calls ORDER BY started_at, rowid")
        assert [(r["tool"], r["capability"], r["status"]) for r in rows[:2]] == [
            ("docs_lookup", "net.docs", "ok"),
            ("web_fetch", "net.http", "ok"),
        ]
        assert [e["via"] for e in events(rt, "research.docs")] == ["tool"]

        # With an approval handler the same request goes through the approval flow and runs.
        async def approve(_approval: Any) -> ApprovalDecision:
            return ApprovalDecision(True)

        rt.approval_handler = approve
        approved = await executor.execute(
            ToolCall("c7", "web_fetch", {"url": "https://example.net/page"}), _tool_context(rt)
        )
        assert approved.ok and "approved external page" in approved.content


async def test_tool_failures_keep_their_error_class(harness: Harness, calc_repo: Path) -> None:
    web = FakeWeb()
    web.route(C7, SEARCH, httpx.Response(429, json={"error": "rate_limited"}, headers={"retry-after": "30"}))
    async with research(harness, web, calc_repo) as (rt, svc):
        executor = ToolExecutor({t.name: t for t in svc.tools_for_role("reviewer")})
        result = await executor.execute(
            ToolCall("c1", "docs_lookup", {"library": "fastapi", "query": "q"}), _tool_context(rt)
        )
    assert not result.ok and result.error_class == "rate_limited"
    assert "retry after 30s" in result.content


async def test_web_fetch_offset_pages_through_long_text(harness: Harness, calc_repo: Path) -> None:
    paragraphs = "".join(f"<p>paragraph {i:03d} {'z' * 200}</p>" for i in range(150))
    web = FakeWeb()
    web.html("docs.python.org", "/long.html", f"<body>{paragraphs}</body>")
    async with research(harness, web, calc_repo) as (rt, svc):
        executor = ToolExecutor({t.name: t for t in svc.tools_for_role("researcher")})
        ctx = _tool_context(rt)
        first = await executor.execute(
            ToolCall("c1", "web_fetch", {"url": "https://docs.python.org/long.html"}), ctx
        )
        assert first.ok and first.data is not None
        offset = first.data["next_offset"]
        assert offset and f"offset={offset}" in first.content and "paragraph 149" not in first.content
        rest = await executor.execute(
            ToolCall("c2", "web_fetch", {"url": "https://docs.python.org/long.html", "offset": offset}), ctx
        )
        assert rest.ok and "paragraph 149" in rest.content and "(cached)" in rest.content
    assert web.paths() == ["/long.html"]


async def test_check_probes_context7_and_reports_failures(harness: Harness) -> None:
    web = FakeWeb().context7()
    async with research(harness, web) as (_rt, svc):
        ready = await svc.check()
        assert ready["ready"] is True and ready["results"] == 2
        web.route(C7, SEARCH, httpx.Response(503, json={"error": "search_failed", "message": "later"}))
        down = await svc.check()
    assert down["ready"] is False and down["error_class"] == "upstream_error"
