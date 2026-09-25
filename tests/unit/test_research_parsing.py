"""Pure research helpers: HTML extraction, URL guards, HTTP header parsing, Context7 shapes."""

from __future__ import annotations

import httpx
import pytest

from coremain.errors import ExitCode, PolicyDeniedError
from coremain.research.context7 import (
    SNIPPET_SEPARATOR,
    bound_docs_text,
    extract_sources,
    library_page_url,
    looks_like_library_id,
    rank_candidates,
    render_json_context,
)
from coremain.research.errors import ResearchError
from coremain.research.htmltext import html_to_text
from coremain.research.http import (
    HttpResponse,
    classify_content,
    decode_body,
    is_local_host,
    is_private_host,
    parse_retry_after,
    validate_url,
)

DOC_PAGE = """<!doctype html>
<html><head>
  <title>Guide &amp; Reference</title>
  <style>body { color: red }</style>
  <script>window.tracker = "should never appear";</script>
</head>
<body>
  <header><a href="/">Logo</a><nav><a href="/docs">Docs</a> <a href="/blog">Blog menu</a></nav></header>
  <div class="cookie-banner">We use cookies</div>
  <main>
    <h1>Install <a class="headerlink" href="#install">¶</a></h1>
    <p>Run <code>pip install demo</code> &mdash; then import it.<br>Second line.</p>
    <ul><li>fast</li><li>typed<ul><li>nested</li></ul></li></ul>
    <ol start="3"><li>third</li><li>fourth</li></ol>
    <pre class="highlight"><code class="language-python">import demo

print(demo.run(&quot;x&quot;))</code></pre>
    <table><tr><th>Name</th><th>Type</th></tr><tr><td>timeout</td><td>float</td></tr></table>
    <blockquote><p>Quoted advice</p></blockquote>
    <aside class="note"><p>Callouts inside main content are kept.</p></aside>
    <div aria-hidden="true">invisible chrome</div>
    <div style="display: none">hidden text</div>
    <p hidden>also hidden</p>
  </main>
  <aside id="sidebar">Sidebar links</aside>
  <footer>© 2026 Footer text</footer>
  <noscript>Enable JavaScript</noscript>
</body></html>
"""


def test_html_extraction_keeps_structure_and_drops_noise() -> None:
    page = html_to_text(DOC_PAGE, "https://docs.example.com/guide/")
    text = page.text
    assert page.title == "Guide & Reference"
    assert page.used_main
    assert text.startswith("# Install")
    assert "¶" not in text
    assert "Run `pip install demo` — then import it.\nSecond line." in text
    assert "- fast\n- typed\n  - nested" in text
    assert "3. third\n4. fourth" in text
    assert '```python\nimport demo\n\nprint(demo.run("x"))\n```' in text
    assert "Name | Type\ntimeout | float" in text
    assert "> Quoted advice" in text
    assert "Callouts inside main content are kept." in text
    for noise in (
        "should never appear",
        "color: red",
        "Blog menu",
        "Logo",
        "cookies",
        "invisible chrome",
        "hidden text",
        "also hidden",
        "Sidebar links",
        "Footer text",
        "Enable JavaScript",
    ):
        assert noise not in text, noise


def test_html_prefers_the_article_inside_a_chrome_heavy_main() -> None:
    listing = "".join(f"<div class='row'>file_{i}.py last commit message</div>" for i in range(40))
    readme = "<h1>Project</h1><p>" + "Readme body text. " * 20 + "</p>"
    page = html_to_text(
        f"<body><main>{listing}<article class='markdown-body'>{readme}</article></main></body>"
    )
    assert page.text.startswith("# Project")
    assert "file_3.py" not in page.text


def test_html_without_content_root_uses_the_whole_body() -> None:
    page = html_to_text("<body><h2>Title</h2><p>Just a paragraph.</p><nav>skip me</nav></body>")
    assert page.text == "## Title\n\nJust a paragraph."
    assert not page.used_main


def test_html_tolerates_malformed_markup() -> None:
    markup = (
        "<div><p>first<p>second</span></div></em><ul><li>one<li>two</ul>"
        "<table><tr><td>a<td>b</table><pre><code>unterminated block"
    )
    text = html_to_text(markup).text
    assert "first\n\nsecond" in text
    assert "- one\n- two" in text
    assert "a | b" in text
    assert text.endswith("```\nunterminated block\n```")
    assert html_to_text("<p>ok</p><script>never closed <b>not markup").text == "ok"


def test_html_list_items_wrapping_paragraphs_keep_their_markers() -> None:
    text = html_to_text("<ul><li><p>alpha</p></li><li><p>beta</p><p>more</p></li></ul>").text
    assert text == "- alpha\n- beta\n\nmore"


def test_html_pre_containing_backticks_uses_a_longer_fence() -> None:
    text = html_to_text("<pre>use ```md fences``` inside</pre>").text
    assert text == "````\nuse ```md fences``` inside\n````"


def test_html_links_are_absolute_deduplicated_and_http_only() -> None:
    page = html_to_text(
        '<body><p><a href="/a#frag">A</a> <a href="/a">A again</a> <a href="javascript:void(0)">js</a>'
        ' <a href="mailto:x@example.com">mail</a> <a href="https://other.example/b">B</a></p></body>',
        "https://docs.example.com/guide/",
    )
    assert page.links == [
        {"text": "A", "url": "https://docs.example.com/a"},
        {"text": "B", "url": "https://other.example/b"},
    ]


def test_html_title_fallbacks_ignore_svg_titles() -> None:
    assert (
        html_to_text('<head><meta property="og:title" content="OG  Title"></head><p>x</p>').title
        == "OG Title"
    )
    svg_first = "<body><svg><title>icon</title></svg><h1>Heading Title</h1><p>body</p></body>"
    assert html_to_text(svg_first).title == "Heading Title"


# ------------------------------------------------------------------------------------ URLs
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://user:pw@docs.python.org/",
        "no-scheme.example/x",
        "http:///nohost",
        "",
    ],
)
def test_validate_url_rejects_unusable_urls(url: str) -> None:
    with pytest.raises(ResearchError) as exc:
        validate_url(url)
    assert exc.value.error_class == "invalid_url"
    assert exc.value.exit_code == ExitCode.USAGE


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://[fe80::1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://2852039166/",
        "http://0xa9fea9fe/",
        "http://100.100.100.200/latest/meta-data/",
    ],
)
def test_validate_url_refuses_link_local_and_metadata_endpoints(url: str) -> None:
    with pytest.raises(PolicyDeniedError):
        validate_url(url)


def test_validate_url_normalizes_and_drops_fragments() -> None:
    assert validate_url("  HTTPS://Docs.Python.org/3/?q=1#section ") == "https://docs.python.org/3/?q=1"
    assert validate_url("http://localhost:8000") == "http://localhost:8000/"
    assert is_local_host("localhost") and is_local_host("127.0.0.2") and is_local_host("[::1]")
    assert not is_local_host("docs.python.org")
    assert all(is_private_host(h) for h in ("localhost", "10.1.2.3", "192.168.0.10", "[fd12::1]", "0.0.0.0"))
    assert not any(is_private_host(h) for h in ("docs.python.org", "8.8.8.8", "[2606:4700::1111]"))


# ------------------------------------------------------------------------------ headers
def test_parse_retry_after_supports_seconds_http_dates_and_ratelimit_reset() -> None:
    now = 1_790_000_000.0
    assert parse_retry_after({"retry-after": "17"}, now) == 17
    assert parse_retry_after({"retry-after": "Mon, 21 Sep 2026 14:13:40 GMT"}, now) == 20
    assert parse_retry_after({"ratelimit-reset": str(int(now) + 90)}, now) == 90
    assert parse_retry_after({"ratelimit-reset": "30"}, now) == 30
    assert parse_retry_after({}, now) is None
    assert parse_retry_after({"retry-after": "soon"}, now) is None


def _response(body: bytes, content_type: str | None) -> HttpResponse:
    headers = httpx.Headers({"content-type": content_type} if content_type else {})
    return HttpResponse("https://x.example/", 200, "OK", headers, body, False)


def test_decode_body_honours_charset_meta_and_bom() -> None:
    assert decode_body(_response("café".encode("latin-1"), "text/plain; charset=ISO-8859-1")) == "café"
    meta = b'<html><head><meta charset="windows-1252"></head><body>\x93quoted\x94</body></html>'
    assert "\u201cquoted\u201d" in decode_body(_response(meta, "text/html"))
    assert decode_body(_response(b"\xef\xbb\xbfhello", "text/plain")) == "hello"
    assert decode_body(_response("hi".encode("utf-16"), None)) == "hi"
    assert decode_body(_response(b"ok", "text/plain; charset=unknown-codec")) == "ok"


def test_classify_content() -> None:
    assert classify_content("text/html", b"") == "html"
    assert classify_content("application/json", b"{}") == "text"
    assert classify_content("application/vnd.api+json", b"{}") == "text"
    assert classify_content("image/png", b"\x89PNG") == "binary"
    assert classify_content("", b"<!DOCTYPE html><p>x") == "html"
    assert classify_content("application/octet-stream", b"\x00\x01binary") == "binary"
    assert classify_content("", "plain ünïcode text".encode()) == "text"


# ------------------------------------------------------------------------------ Context7
def test_library_id_detection_and_page_url() -> None:
    assert looks_like_library_id("/vercel/next.js")
    assert looks_like_library_id("/vercel/next.js/v15.1.8")
    assert looks_like_library_id("/vercel/next.js@v15.1.8")
    assert not looks_like_library_id("next.js")
    assert not looks_like_library_id("/single")
    assert not looks_like_library_id("/org/ignore previous instructions")
    assert not looks_like_library_id("/org/" + "a" * 300)
    assert (
        library_page_url("https://context7.com/api/", "/vercel/next.js")
        == "https://context7.com/vercel/next.js"
    )


def test_rank_candidates_prefers_exact_name_then_context7_order_and_penalizes_unfinished() -> None:
    results = [
        {
            "id": "/kludex/fastapi-tips",
            "title": "FastAPI Tips",
            "totalSnippets": 44,
            "trustScore": 10,
            "state": "finalized",
        },
        {
            "id": "/websites/fastapi_tiangolo",
            "title": "FastAPI",
            "totalSnippets": 2377,
            "trustScore": 9,
            "benchmarkScore": 88,
            "state": "finalized",
        },
        {
            "id": "/fastapi/fastapi",
            "title": "FastAPI",
            "totalSnippets": 3000,
            "trustScore": 9,
            "state": "initial",
        },
        {"id": "not-an-id", "title": "FastAPI"},
        "garbage",
    ]
    ranked = rank_candidates("FastAPI", [r for r in results if isinstance(r, dict)])
    assert [c.id for c in ranked] == [
        "/websites/fastapi_tiangolo",
        "/kludex/fastapi-tips",
        "/fastapi/fastapi",
    ]
    assert "exact name match" in ranked[0].reasons
    assert "state initial" in ranked[2].reasons
    assert ranked[0].to_dict(selected=True)["context7_rank"] == 2


def test_render_json_context_matches_the_text_layout() -> None:
    text = render_json_context(
        {
            "codeSnippets": [
                {
                    "codeTitle": "Path Parameters",
                    "codeDescription": "Declare path parameters.",
                    "codeId": "https://fastapi.tiangolo.com/tutorial/path-params/",
                    "codeList": [{"language": "python", "code": "@app.get('/items/{item_id}')"}],
                }
            ],
            "infoSnippets": [
                {
                    "breadcrumb": "Tutorial > Path",
                    "pageId": "https://fastapi.tiangolo.com/tutorial/",
                    "content": "Info.",
                }
            ],
        }
    )
    assert text.startswith(
        "### Path Parameters\n\nSource: https://fastapi.tiangolo.com/tutorial/path-params/"
    )
    assert "```python\n@app.get('/items/{item_id}')\n```" in text
    assert f"\n\n{SNIPPET_SEPARATOR}\n\n### Tutorial > Path" in text
    assert extract_sources(text) == [
        "https://fastapi.tiangolo.com/tutorial/path-params/",
        "https://fastapi.tiangolo.com/tutorial/",
    ]
    with pytest.raises(ResearchError):
        render_json_context(["not", "an", "object"])


def test_bound_docs_text_cuts_between_snippets() -> None:
    snippets = [f"### S{i}\n\nSource: https://example.com/{i}\n\n" + "x" * 1000 for i in range(12)]
    text = f"\n\n{SNIPPET_SEPARATOR}\n\n".join(snippets)
    bounded, truncated, total, kept = bound_docs_text(text, 5000)
    assert truncated and total == 12 and kept == 4
    assert len(bounded) <= 5000
    assert bounded.count("x" * 1000) == kept
    assert "8 more snippet(s) omitted" in bounded
    single, truncated, total, kept = bound_docs_text("y" * 9000, 5000)
    assert truncated and total == 1 and kept == 1 and len(single) <= 5000
    assert bound_docs_text("short", 5000) == ("short", False, 1, 1)
