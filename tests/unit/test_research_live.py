"""Live checks against the real Context7 API and a real documentation page (``-m network``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from coremain.research.errors import ResearchError
from coremain.research.service import ResearchService

if TYPE_CHECKING:
    from tests.helpers import Harness

pytestmark = pytest.mark.network


@pytest.fixture(scope="module", autouse=True)
def _reachable() -> None:
    """Skip (rather than fail) when the machine has no route to the live endpoints."""
    for url in ("https://context7.com/api/v2/libs/search", "https://docs.python.org/3/"):
        try:
            httpx.head(url, timeout=8.0)
        except httpx.TransportError as exc:
            pytest.skip(f"network unavailable ({url}: {type(exc).__name__})")


async def test_live_context7_lookup_resolves_caches_and_follows_library_redirects(harness: Harness) -> None:
    async with harness.runtime() as rt:
        service = rt.extension("research")
        assert isinstance(service, ResearchService)
        try:
            result = await service.docs("fastapi", "dependencies with yield")
            renamed = await service.docs("/fastapi/fastapi", "path parameters")
        except ResearchError as exc:
            if exc.error_class == "rate_limited":
                pytest.skip(f"Context7 anonymous rate limit reached: {exc.message}")
            raise
        assert result["source"] == "context7" and result["cached"] is False
        assert result["library_id"].startswith("/") and result["library_id"].count("/") >= 2
        assert result["candidates"] and result["candidates"][0]["selected"] is True
        assert "exact name match" in result["selection_reason"]
        assert "yield" in result["text"] and result["sources"]
        assert (await service.docs("fastapi", "dependencies with yield"))["cached"] is True
        # Context7 answers renamed libraries with HTTP 301 + JSON redirectUrl; either way we land on docs.
        assert renamed["text"]
        if renamed["library_id"] != "/fastapi/fastapi":
            assert renamed["redirected_from"] == "/fastapi/fastapi"


async def test_live_web_fetch_extracts_python_docs(harness: Harness) -> None:
    async with harness.runtime() as rt:
        page = await rt.extension("research").fetch("https://docs.python.org/3/library/html.parser.html")
    assert page["status"] == 200 and page["content_type"] == "text/html"
    assert page["title"] and "html.parser" in page["title"]
    assert "class html.parser.HTMLParser" in page["text"]
    assert "¶" not in page["text"]
    assert page["bytes"] > 10_000 and not page["truncated"]
