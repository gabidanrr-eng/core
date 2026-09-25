"""Context7 documentation client.

Verified live against https://context7.com/api on 2026-09-25 (anonymous and with a bogus key) and
cross-checked with the published API guide (https://context7.com/docs/api-guide):

* ``GET {base}/v2/libs/search?libraryName=<name>&query=<question>`` → ``200 application/json``
  ``{"results": [{"id": "/org/project", "title", "description", "branch", "lastUpdateDate",
  "state": "finalized", "totalTokens", "totalSnippets", "stars", "trustScore" (0-10),
  "benchmarkScore" (0-100), "versions": [...]}], "searchFilterApplied": bool}``, already ranked by
  Context7 for the question.
* ``GET {base}/v2/context?libraryId=/org/project&query=<question>`` → ``200 text/plain`` snippets
  (``### title`` / ``Source: <url>`` / description / fenced code) separated by lines of dashes.
  ``type=json`` instead yields ``{"codeSnippets": [{"codeTitle", "codeDescription", "codeLanguage",
  "codeTokens", "codeId" (source URL), "pageTitle", "codeList": [{"language", "code"}]}],
  "infoSnippets": [{"pageId" (source URL), "breadcrumb", "content", "contentTokens"}]}``; both
  shapes are accepted.
* Errors are JSON ``{"error": <code>, "message": <text>}``: 400 ``validation_error`` (e.g. missing
  query), 401 ``invalid_api_key`` (keys start with ``ctx7sk``), 404 ``library_not_found``, 429 with
  ``Retry-After``; renamed libraries answer **301** with JSON ``redirectUrl`` holding the new library
  id (e.g. ``/fastapi/fastapi`` → ``/websites/fastapi_tiangolo``), not an HTTP ``Location``; 202
  means the library is still being processed.
* Authentication is optional: ``Authorization: Bearer <key>``. Anonymous responses carry
  ``context7-quota-tier: anonymous`` and ``RateLimit-Limit/-Remaining/-Reset`` (Unix time) headers.
* The legacy ``/v1/search?query=`` and ``/v1/<id>?type=txt&topic=`` forms still respond (v1 search is
  rewritten server-side to ``/v2/search``) but are not used here.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from coremain.research.errors import ResearchError
from coremain.research.http import HttpResponse, bounded_get, parse_retry_after
from coremain.util.text import one_line

SEARCH_PATH = "/v2/libs/search"
CONTEXT_PATH = "/v2/context"
LIBRARY_ID_RE = re.compile(r"^/[\w.@+~-]+(?:/[\w.@+~-]+)+$")
MAX_QUERY_CHARS = 500
MAX_LIBRARY_CHARS = 200
DOCS_MAX_CHARS = 20_000
MAX_API_BYTES = 4_000_000
MAX_CANDIDATES = 5
MAX_LIBRARY_REDIRECTS = 3
SNIPPET_SEPARATOR = "-" * 32
_SEPARATOR_RE = re.compile(r"\n\s*-{16,}\s*\n")
_SOURCE_RE = re.compile(r"^Source:\s*(\S+)", re.MULTILINE)


def looks_like_library_id(value: str) -> bool:
    value = value.strip()
    return len(value) <= MAX_LIBRARY_CHARS and bool(LIBRARY_ID_RE.match(value))


def library_page_url(base_url: str, library_id: str) -> str:
    """Context7 library ids are the URL path of the library page on the site."""
    root = base_url.rstrip("/")
    if root.endswith("/api"):
        root = root[: -len("/api")]
    return root + library_id


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        return None
    return float(value)


@dataclass
class Candidate:
    id: str
    title: str
    description: str
    snippets: int
    trust_score: float | None
    benchmark_score: float | None
    state: str | None
    versions: list[str]
    rank: int
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self, *, selected: bool) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": one_line(self.description, 200),
            "snippets": self.snippets,
            "trust_score": self.trust_score,
            "benchmark_score": self.benchmark_score,
            "state": self.state,
            "versions": self.versions,
            "context7_rank": self.rank,
            "score": round(self.score, 3),
            "selected": selected,
        }


def rank_candidates(library: str, results: list[dict[str, Any]]) -> list[Candidate]:
    """Deterministic ranking: name match first, then Context7's own order, trust, coverage, benchmark."""
    parsed: list[Candidate] = []
    for item in results:
        lib_id = item.get("id")
        if not isinstance(lib_id, str) or not looks_like_library_id(lib_id):
            continue
        snippets = _num(item.get("totalSnippets"))
        versions = item.get("versions")
        parsed.append(
            Candidate(
                id=lib_id,
                title=str(item.get("title") or lib_id),
                description=str(item.get("description") or ""),
                snippets=int(snippets) if snippets and snippets > 0 else 0,
                trust_score=_num(item.get("trustScore")),
                benchmark_score=_num(item.get("benchmarkScore")),
                state=item.get("state") if isinstance(item.get("state"), str) else None,
                versions=[v for v in versions if isinstance(v, str)][:8]
                if isinstance(versions, list)
                else [],
                rank=len(parsed) + 1,
            )
        )
    total = len(parsed)
    target = _norm(library)
    for c in parsed:
        title = _norm(c.title)
        segments = [_norm(s) for s in c.id.strip("/").split("/")]
        if target and (title == target or target in segments):
            c.score += 3.0
            c.reasons.append("exact name match")
        elif target and (title.startswith(target) or any(s.startswith(target) for s in segments)):
            c.score += 1.5
            c.reasons.append("name prefix match")
        elif target and (target in title or any(target in s for s in segments)):
            c.score += 1.0
            c.reasons.append("partial name match")
        c.score += 2.0 * (total - c.rank + 1) / total
        c.reasons.append(f"Context7 rank #{c.rank} of {total}")
        if c.trust_score is not None and c.trust_score > 0:
            c.score += min(c.trust_score, 10.0) / 10.0
            c.reasons.append(f"trust {c.trust_score:g}/10")
        c.score += min(1.0, math.log10(1 + c.snippets) / 4)
        c.reasons.append(f"{c.snippets} snippets")
        if c.benchmark_score is not None and c.benchmark_score > 0:
            c.score += 0.8 * min(c.benchmark_score, 100.0) / 100.0
        if c.state not in (None, "finalized"):
            c.score -= 4.0
            c.reasons.append(f"state {c.state}")
    return sorted(parsed, key=lambda c: (-c.score, c.rank))


def render_json_context(payload: Any) -> str:
    """Render the ``type=json`` context shape in the same layout as the plain-text format."""
    if not isinstance(payload, dict):
        raise ResearchError(
            "Context7 returned an unexpected JSON context shape", error_class="invalid_response"
        )
    parts: list[str] = []
    for snip in payload.get("codeSnippets") or []:
        if not isinstance(snip, dict):
            continue
        lines = [f"### {snip.get('codeTitle') or 'Snippet'}", ""]
        if snip.get("codeId"):
            lines += [f"Source: {snip['codeId']}", ""]
        if snip.get("codeDescription"):
            lines += [str(snip["codeDescription"]), ""]
        for code in snip.get("codeList") or []:
            if isinstance(code, dict) and code.get("code"):
                lines += [f"```{code.get('language') or ''}".rstrip(), str(code["code"]).rstrip(), "```", ""]
        parts.append("\n".join(lines).strip())
    for info in payload.get("infoSnippets") or []:
        if not isinstance(info, dict) or not info.get("content"):
            continue
        lines = [f"### {info.get('breadcrumb') or 'Documentation'}", ""]
        if info.get("pageId"):
            lines += [f"Source: {info['pageId']}", ""]
        lines.append(str(info["content"]).strip())
        parts.append("\n".join(lines).strip())
    return f"\n\n{SNIPPET_SEPARATOR}\n\n".join(parts)


def bound_docs_text(text: str, limit: int = DOCS_MAX_CHARS) -> tuple[str, bool, int, int]:
    """Cut at snippet boundaries to at most ``limit`` chars; returns (text, truncated, total, kept)."""
    text = text.strip()
    snippets = [s.strip() for s in _SEPARATOR_RE.split(text) if s.strip()]
    total = len(snippets)
    if len(text) <= limit:
        return text, False, total, total
    joiner = f"\n\n{SNIPPET_SEPARATOR}\n\n"
    budget = limit - 200
    kept: list[str] = []
    size = 0
    for snip in snippets:
        add = len(snip) + (len(joiner) if kept else 0)
        if size + add > budget:
            break
        kept.append(snip)
        size += add
    if not kept:
        kept = [snippets[0][:budget].rstrip() + "\n… [snippet truncated]"] if snippets else []
    omitted = total - len(kept)
    body = joiner.join(kept)
    if omitted:
        body += f"\n\n[… {omitted} more snippet(s) omitted to stay within the context budget; ask a narrower query …]"
    return body, True, total, len(kept)


def extract_sources(text: str, limit: int = 20) -> list[str]:
    seen: list[str] = []
    for match in _SOURCE_RE.finditer(text):
        url = match.group(1)
        if url not in seen:
            seen.append(url)
            if len(seen) >= limit:
                break
    return seen


def rate_limit_info(headers: httpx.Headers) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for header, name in (
        ("ratelimit-limit", "limit"),
        ("ratelimit-remaining", "remaining"),
        ("ratelimit-reset", "reset_at"),
    ):
        value = (headers.get(header) or "").strip()
        if value.isdigit():
            out[name] = int(value)
    tier = headers.get("context7-quota-tier")
    if tier:
        out["tier"] = tier[:40]
    return out or None


@dataclass
class ContextResult:
    library_id: str
    text: str
    redirected_from: str | None
    rate_limit: dict[str, Any] | None


class Context7Client:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str | None,
        *,
        redact: Callable[[str], str],
        now: Callable[[], float],
        retries: int = 1,
        backoff_s: float = 0.5,
    ):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.redact = redact
        self.now = now
        self.retries = retries
        self.backoff_s = backoff_s

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/plain, application/json;q=0.9"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get(self, path: str, params: dict[str, str]) -> HttpResponse:
        return await bounded_get(
            self.client,
            self.base_url + path,
            params=params,
            headers=self._headers(),
            max_bytes=MAX_API_BYTES,
            retries=self.retries,
            backoff_s=self.backoff_s,
        )

    async def search(self, library: str, query: str) -> list[dict[str, Any]]:
        res = await self._get(SEARCH_PATH, {"libraryName": library, "query": query})
        if res.status != 200:
            raise self.error(res, f"library search for '{library}'")
        try:
            payload = res.json()
        except ValueError as exc:
            raise ResearchError(
                "Context7 returned a malformed search response (not JSON)", error_class="invalid_response"
            ) from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise ResearchError(
                "Context7 search response has no 'results' list", error_class="invalid_response"
            )
        return [r for r in results if isinstance(r, dict)]

    async def context(self, library_id: str, query: str) -> ContextResult:
        current = library_id
        redirected_from: str | None = None
        for _ in range(MAX_LIBRARY_REDIRECTS + 1):
            res = await self._get(CONTEXT_PATH, {"libraryId": current, "query": query})
            if res.status == 301:
                target = self._redirect_target(res)
                if target is None or target == current:
                    raise ResearchError(
                        f"Context7 redirected {current} without a usable library id",
                        error_class="invalid_response",
                        status=301,
                    )
                redirected_from = redirected_from or library_id
                current = target
                continue
            if res.status != 200:
                raise self.error(res, f"documentation request for {current}")
            return ContextResult(current, self._body_text(res), redirected_from, rate_limit_info(res.headers))
        raise ResearchError(
            f"Context7 library redirects for {library_id} did not settle after {MAX_LIBRARY_REDIRECTS} hops",
            error_class="invalid_response",
        )

    @staticmethod
    def _redirect_target(res: HttpResponse) -> str | None:
        try:
            payload = res.json()
        except ValueError:
            return None
        target = payload.get("redirectUrl") if isinstance(payload, dict) else None
        return target.strip() if isinstance(target, str) and looks_like_library_id(target) else None

    @staticmethod
    def _body_text(res: HttpResponse) -> str:
        raw = res.text()
        ctype = res.content_type
        if (
            ctype == "application/json"
            or ctype.endswith("+json")
            or (not ctype and raw.lstrip().startswith("{"))
        ):
            try:
                payload = json.loads(raw)
            except ValueError as exc:
                raise ResearchError(
                    "Context7 returned malformed JSON documentation", error_class="invalid_response"
                ) from exc
            if isinstance(payload, dict) and "error" in payload and "codeSnippets" not in payload:
                raise ResearchError(
                    f"Context7 returned an error document: {one_line(str(payload.get('message') or payload['error']), 300)}",
                    error_class="invalid_response",
                )
            return render_json_context(payload)
        return raw

    def error(self, res: HttpResponse, what: str) -> ResearchError:
        code, message = _error_body(res)
        status = res.status
        label = f"HTTP {status}" + (f" {code}" if code else "")
        detail = f": {self.redact(one_line(message, 300))}" if message else ""
        if status in {401, 403}:
            hint = (
                "check the key referenced by research.context7_api_key (Context7 keys start with 'ctx7sk')"
                if self.api_key
                else "this library or endpoint requires a Context7 API key; set research.context7_api_key "
                "to a credential reference such as env:CONTEXT7_API_KEY"
            )
            return ResearchError(
                f"Context7 rejected the {what} ({label}){detail}",
                error_class="auth_failed",
                status=status,
                hint=hint,
            )
        if status == 404:
            return ResearchError(
                f"Context7 found nothing for the {what} ({label}){detail}",
                error_class="not_found",
                status=status,
                hint="search with a plain library name, or copy the exact id from the library page on context7.com",
            )
        if status == 429:
            retry = parse_retry_after(res.headers, self.now())
            return ResearchError(
                f"Context7 rate limit exceeded ({label})"
                + (f"; retry after {retry}s" if retry is not None else ""),
                error_class="rate_limited",
                status=status,
                retry_after_s=retry,
                hint=None
                if self.api_key
                else "anonymous access has low limits; set research.context7_api_key for higher ones",
            )
        if status == 202:
            return ResearchError(
                f"Context7 is still processing this library ({label}); retry later",
                error_class="not_ready",
                status=status,
            )
        if status in {400, 422}:
            return ResearchError(
                f"Context7 rejected the {what} as invalid ({label}){detail}",
                error_class="invalid_request",
                status=status,
            )
        if status >= 500:
            return ResearchError(
                f"Context7 upstream error for the {what} ({label}){detail}",
                error_class="upstream_error",
                status=status,
            )
        return ResearchError(
            f"unexpected Context7 response for the {what} ({label}){detail}",
            error_class="upstream_error",
            status=status,
        )


def _error_body(res: HttpResponse) -> tuple[str | None, str | None]:
    try:
        payload = res.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        code = payload.get("error")
        message = payload.get("message")
        return (
            code if isinstance(code, str) else None,
            message if isinstance(message, str) else (code if isinstance(code, str) else None),
        )
    if res.content_type == "text/plain":
        text = res.text().strip()
        return None, text or None
    return None, None
