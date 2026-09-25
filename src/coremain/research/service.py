"""ResearchService: policy-governed documentation retrieval (Context7) and web fetch.

Authorization:

* Direct user commands (``ctx=None``, e.g. ``core research docs``) evaluate the policy engine here:
  ``deny`` refuses; ``ask`` proceeds because the user initiated the request.
* Tool calls (``ctx`` given) were already evaluated by ``ToolExecutor`` for their initial target.
  Redirect hops to other URLs are re-evaluated here and refused unless the policy allows them
  outright, since no approval can be requested mid-request; the model can ask for the redirect
  target explicitly instead.

Results are redacted before they are returned or cached. The cache lives in ``cache_entries``
(namespaces ``research.docs``, ``research.resolve``, ``research.fetch``) with ``research.cache_ttl_s``
expiry; local targets and ``Cache-Control: no-store`` pages are never cached. In offline mode no
network request is made: cached results are served only if policy would allow the request online.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx

from coremain.config.schema import ResearchConfig
from coremain.errors import CapabilityUnavailable, CoreError, PolicyDeniedError, UsageError
from coremain.research.cache import ResearchCache
from coremain.research.context7 import (
    DOCS_MAX_CHARS,
    MAX_CANDIDATES,
    MAX_LIBRARY_CHARS,
    MAX_QUERY_CHARS,
    Context7Client,
    bound_docs_text,
    extract_sources,
    library_page_url,
    looks_like_library_id,
    rank_candidates,
)
from coremain.research.errors import ResearchError
from coremain.research.htmltext import html_to_text
from coremain.research.http import (
    bounded_get,
    classify_content,
    decode_body,
    host_of,
    is_local_host,
    is_private_host,
    validate_url,
    web_status_error,
)
from coremain.research.tools import DocsLookup, WebFetch
from coremain.security.credentials import resolve_credential
from coremain.security.policy import Capability, PolicyEngine, PolicyRequest
from coremain.tools.base import Tool
from coremain.util.jsonutil import stable_hash
from coremain.util.text import one_line
from coremain.version import __version__

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime
    from coremain.tools.base import ToolContext

NS_DOCS = "research.docs"
NS_RESOLVE = "research.resolve"
NS_FETCH = "research.fetch"
DOCS_ROLES = frozenset({"planner", "implementer", "corrector", "debugger", "researcher", "reviewer"})
FETCH_ROLES = frozenset({"planner", "implementer", "corrector", "debugger", "researcher"})
MAX_REDIRECTS = 5
FETCH_TEXT_MAX_CHARS = 100_000
REQUEST_DEADLINE_S = 120.0
WEB_ACCEPT = "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.5"


class ResearchService:
    retries = 1
    retry_backoff_s = 0.5

    def __init__(self, rt: CoreRuntime, *, transport: httpx.AsyncBaseTransport | None = None):
        self.rt = rt
        self.cache = ResearchCache(rt.db, rt.clock)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._key: tuple[str, str] | None = None

    # ------------------------------------------------------------------ config
    @property
    def config(self) -> ResearchConfig:
        return self.rt.config.research

    @property
    def offline(self) -> bool:
        return self.rt.config.permissions.network.offline

    @property
    def docs_base_url(self) -> str:
        return self.config.context7_base_url.rstrip("/")

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
                headers={"user-agent": f"core-main/{__version__} (research)"},
                follow_redirects=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------- tools
    def tools_for_role(self, role: str) -> list[Tool]:
        if self.offline:
            return []
        tools: list[Tool] = []
        if role in DOCS_ROLES and self.config.docs_provider != "none":
            tools.append(DocsLookup(self))
        if role in FETCH_ROLES:
            tools.append(WebFetch(self))
        return tools

    # -------------------------------------------------------------------- docs
    async def docs(self, library: str, query: str, *, ctx: ToolContext | None = None) -> dict[str, Any]:
        started = time.monotonic()
        try:
            async with asyncio.timeout(REQUEST_DEADLINE_S):
                result = await self._docs(library, query, ctx)
        except TimeoutError as exc:
            err = ResearchError(
                f"documentation lookup exceeded {REQUEST_DEADLINE_S:.0f}s", error_class="timeout"
            )
            self._emit("research.docs", ctx, started, err, library=library)
            raise err from exc
        except CoreError as exc:
            self._emit("research.docs", ctx, started, exc, library=library)
            raise
        self._emit(
            "research.docs",
            ctx,
            started,
            None,
            library=library,
            library_id=result["library_id"],
            cached=result["cached"],
            chars=len(result["text"]),
        )
        return result

    async def _docs(self, library: str, query: str, ctx: ToolContext | None) -> dict[str, Any]:
        cfg = self.config
        if cfg.docs_provider == "none":
            raise CapabilityUnavailable(
                'documentation retrieval is disabled (research.docs_provider = "none")',
                hint='set research.docs_provider = "context7" to enable it',
            )
        library = " ".join(library.split())
        if not library:
            raise UsageError("a library name or Context7 id is required")
        if len(library) > MAX_LIBRARY_CHARS:
            raise UsageError(f"library must be at most {MAX_LIBRARY_CHARS} characters")
        query = " ".join(query.split())[:MAX_QUERY_CHARS] or library
        base = self.docs_base_url
        direct = looks_like_library_id(library)
        key = stable_hash(
            {
                "v": 1,
                "base": base,
                "library": library if direct else library.lower(),
                "query": query,
                "credential": cfg.context7_api_key,
            }
        )
        if self.offline and not is_local_host(host_of(base)):
            return self._offline_hit(NS_DOCS, key, Capability.NET_DOCS, base, ctx, "documentation lookup")
        if ctx is None:
            self._authorize_user(Capability.NET_DOCS, base, "research.docs")
        cached = self.cache.get(NS_DOCS, key)
        if cached is not None:
            return {**self.rt.redactor.redact_obj(cached), "cached": True}
        api = Context7Client(
            self.client(),
            base,
            await self._context7_key(),
            redact=self.rt.redactor.redact,
            now=self.rt.clock.now,
            retries=self.retries,
            backoff_s=self.retry_backoff_s,
        )
        candidates: list[dict[str, Any]] = []
        if direct:
            library_id, reason = library, "explicit Context7 library id"
        else:
            library_id, reason, candidates = await self._resolve(api, library, query)
        found = await api.context(library_id, query)
        text, truncated, total, kept = bound_docs_text(self.rt.redactor.redact(found.text), DOCS_MAX_CHARS)
        if not text:
            raise ResearchError(
                f"Context7 returned no documentation for {found.library_id} matching '{one_line(query, 120)}'",
                error_class="not_found",
                hint="try a broader question",
            )
        result: dict[str, Any] = {
            "library_id": found.library_id,
            "source": "context7",
            "query": query,
            "text": text,
            "url": library_page_url(base, found.library_id),
            "sources": extract_sources(text),
            "truncated": truncated,
            "snippets": {"total": total, "included": kept},
            "candidates": candidates,
            "selection_reason": reason,
            "redirected_from": found.redirected_from,
            "fetched_at": self.rt.clock.now(),
        }
        self.cache.put(NS_DOCS, key, result, cfg.cache_ttl_s)
        return {**result, "cached": False, "rate_limit": found.rate_limit}

    async def _resolve(
        self, api: Context7Client, library: str, query: str
    ) -> tuple[str, str, list[dict[str, Any]]]:
        # Resolution is cached per library name (not per question): a name maps to one library.
        key = stable_hash(
            {
                "v": 1,
                "base": api.base_url,
                "library": library.lower(),
                "credential": self.config.context7_api_key,
            }
        )
        hit = self.cache.get(NS_RESOLVE, key)
        if (
            hit is not None
            and isinstance(hit.get("library_id"), str)
            and isinstance(hit.get("candidates"), list)
        ):
            return hit["library_id"], str(hit.get("reason") or ""), hit["candidates"]
        ranked = rank_candidates(library, await api.search(library, query))
        if not ranked:
            raise ResearchError(
                f"no Context7 library matches '{library}'",
                error_class="not_found",
                hint="try another name, or pass an exact Context7 id such as /vercel/next.js",
            )
        best = ranked[0]
        reason = f"best match for '{library}': {best.id} ({best.title}): " + ", ".join(best.reasons)
        candidates = self.rt.redactor.redact_obj(
            [c.to_dict(selected=i == 0) for i, c in enumerate(ranked[:MAX_CANDIDATES])]
        )
        self.cache.put(
            NS_RESOLVE,
            key,
            {"library_id": best.id, "reason": reason, "candidates": candidates},
            self.config.cache_ttl_s,
        )
        return best.id, reason, candidates

    async def _context7_key(self) -> str | None:
        ref = self.config.context7_api_key
        if not ref:
            return None
        if self._key is not None and self._key[0] == ref:
            return self._key[1]
        resolved = await asyncio.to_thread(
            resolve_credential, ref, self.rt.credentials, env=self.rt.env, redactor=self.rt.redactor
        )
        if not resolved.value:
            raise ResearchError(
                f"the Context7 API key reference '{ref}' could not be resolved: {resolved.error or 'empty value'}",
                error_class="credential_missing",
                hint="fix the credential, or unset research.context7_api_key to use anonymous access (lower rate limits)",
            )
        self._key = (ref, resolved.value)
        return resolved.value

    # ------------------------------------------------------------------- fetch
    async def fetch(self, url: str, *, ctx: ToolContext | None = None) -> dict[str, Any]:
        started = time.monotonic()
        try:
            async with asyncio.timeout(REQUEST_DEADLINE_S):
                result = await self._fetch(url, ctx)
        except TimeoutError as exc:
            err = ResearchError(f"web fetch exceeded {REQUEST_DEADLINE_S:.0f}s", error_class="timeout")
            self._emit("research.fetch", ctx, started, err, url=url)
            raise err from exc
        except CoreError as exc:
            self._emit("research.fetch", ctx, started, exc, url=url)
            raise
        self._emit(
            "research.fetch",
            ctx,
            started,
            None,
            url=url,
            final_url=result["final_url"],
            status=result["status"],
            cached=result["cached"],
            bytes=result["bytes"],
            truncated=result["truncated"],
        )
        return result

    async def _fetch(self, url: str, ctx: ToolContext | None) -> dict[str, Any]:
        cfg = self.config
        target = validate_url(url)
        local = is_local_host(host_of(target))
        key = stable_hash({"v": 1, "url": target})
        if self.offline and not local:
            return self._offline_hit(NS_FETCH, key, Capability.NET_HTTP, target, ctx, "web fetch")
        if ctx is None:
            self._authorize_user(Capability.NET_HTTP, target, "research.fetch")
        authorize_hop = self._hop_authorizer(ctx, target)
        cacheable = cfg.cache_ttl_s > 0 and not local
        if cacheable:
            cached = self.cache.get(NS_FETCH, key)
            if cached is not None:
                for hop in cached.get("redirects") or []:
                    authorize_hop(str(hop))
                return {**self.rt.redactor.redact_obj(cached), "cached": True}
        res = await bounded_get(
            self.client(),
            target,
            headers={"accept": WEB_ACCEPT},
            max_bytes=cfg.max_fetch_bytes,
            max_redirects=MAX_REDIRECTS,
            authorize_hop=authorize_hop,
            retries=self.retries,
            backoff_s=self.retry_backoff_s,
        )
        if res.status >= 300:
            raise web_status_error(res, self.rt.clock.now())
        kind = classify_content(res.content_type, res.body)
        if kind == "binary":
            raise ResearchError(
                f"{res.url} returned non-text content ({res.content_type or 'unknown type'}); "
                "web_fetch renders HTML and text only",
                error_class="unsupported_content",
                status=res.status,
            )
        raw = decode_body(res)
        title: str | None = None
        links: list[dict[str, str]] = []
        if kind == "html":
            page = await asyncio.to_thread(html_to_text, raw, res.url)
            title, text, links = page.title, page.text, page.links
        else:
            text = raw.strip()
        redactor = self.rt.redactor
        text = redactor.redact(text)
        text_truncated = len(text) > FETCH_TEXT_MAX_CHARS
        if text_truncated:
            text = (
                text[:FETCH_TEXT_MAX_CHARS].rstrip()
                + f"\n\n[… truncated: {len(text) - FETCH_TEXT_MAX_CHARS} more characters …]"
            )
        result: dict[str, Any] = {
            "url": target,
            "final_url": res.url,
            "status": res.status,
            "content_type": res.content_type or None,
            "title": redactor.redact(title) if title else None,
            "text": text,
            "links": redactor.redact_obj(links),
            "redirects": res.redirects,
            "bytes": len(res.body),
            "truncated": res.truncated or text_truncated,
            "fetched_at": self.rt.clock.now(),
        }
        if cacheable and "no-store" not in res.headers.get("cache-control", "").lower():
            self.cache.put(NS_FETCH, key, result, cfg.cache_ttl_s)
        return {**result, "cached": False}

    # ------------------------------------------------------------------ policy
    def _request(self, capability: str, target: str, ctx: ToolContext | None, tool: str) -> PolicyRequest:
        if ctx is not None:
            return PolicyRequest(
                capability=capability,
                target=target,
                workspace_root=ctx.workspace.path,
                workspace_isolated=ctx.workspace.isolated,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                tool=tool,
            )
        project = self.rt.project
        return PolicyRequest(
            capability=capability, target=target, project_id=project.id if project else None, tool=tool
        )

    def _authorize_user(self, capability: str, target: str, tool: str) -> None:
        decision = self.rt.policy.evaluate(self._request(capability, target, None, tool))
        if decision.decision == "deny":
            raise PolicyDeniedError(
                f"{capability} access to {host_of(target) or target} denied by policy: {decision.reason}",
                details={"target": target, "source": decision.source},
            )

    def _hop_authorizer(self, ctx: ToolContext | None, origin: str) -> Callable[[str], None]:
        public_origin = not is_private_host(host_of(origin))

        def check(target: str) -> None:
            if public_origin and is_private_host(host_of(target)):
                # A public page must not be able to steer requests into local services (SSRF).
                raise PolicyDeniedError(
                    f"redirect from {host_of(origin)} to the local/private address {host_of(target)} is refused",
                    details={"target": target},
                )
            if ctx is None:
                self._authorize_user(Capability.NET_HTTP, target, "research.fetch")
                return
            decision = self.rt.policy.evaluate(self._request(Capability.NET_HTTP, target, ctx, "web_fetch"))
            if decision.decision == "deny":
                raise PolicyDeniedError(
                    f"redirect to {target} denied by policy: {decision.reason}",
                    details={"target": target, "source": decision.source},
                )
            if decision.decision == "ask":
                raise PolicyDeniedError(
                    f"redirect to {target} needs approval ({decision.reason}); "
                    "call web_fetch with that URL directly to request it",
                    details={"target": target, "source": decision.source},
                )

        return check

    def _allowed_online(self, capability: str, target: str, ctx: ToolContext | None) -> bool:
        perms = self.rt.config.permissions
        online = perms.model_copy(update={"network": perms.network.model_copy(update={"offline": False})})
        engine = PolicyEngine(
            online,
            db=self.rt.db,
            clock=self.rt.clock,
            protected_paths=[],
            profile_override=self.rt.policy.profile,
        )
        return engine.evaluate(self._request(capability, target, ctx, "research.offline")).decision != "deny"

    def _offline_hit(
        self, namespace: str, key: str, capability: str, target: str, ctx: ToolContext | None, what: str
    ) -> dict[str, Any]:
        cached = self.cache.get(namespace, key)
        if cached is None:
            raise CapabilityUnavailable(
                f"offline mode: {what} needs network access to {host_of(target)} and no cached copy is available",
                hint="set permissions.network.offline = false, or repeat the request once online to cache it",
            )
        hops = [str(h) for h in cached.get("redirects") or []]
        checks = [(capability, target), *((Capability.NET_HTTP, hop) for hop in hops)]
        if not all(self._allowed_online(cap, tgt, ctx) for cap, tgt in checks):
            raise PolicyDeniedError(
                f"cached {what} result for {host_of(target)} is no longer permitted by policy"
            )
        return {**self.rt.redactor.redact_obj(cached), "cached": True, "offline": True}

    # ------------------------------------------------------------- diagnostics
    def describe(self) -> dict[str, Any]:
        """Configuration and cache state without network access or secret values."""
        cfg = self.config
        if cfg.context7_api_key:
            credential = resolve_credential(
                cfg.context7_api_key, self.rt.credentials, env=self.rt.env, redactor=self.rt.redactor
            ).describe()
        else:
            credential = "none (anonymous Context7 access, lower rate limits)"
        if self.offline:
            status = "offline"
        elif cfg.docs_provider == "none":
            status = "docs disabled"
        else:
            status = "configured"
        return {
            "status": status,
            "docs_provider": cfg.docs_provider,
            "docs_base_url": self.docs_base_url if cfg.docs_provider != "none" else None,
            "credential": credential,
            "offline": self.offline,
            "max_fetch_bytes": cfg.max_fetch_bytes,
            "cache_ttl_s": cfg.cache_ttl_s,
            "cache": self.cache.stats(),
        }

    async def check(self) -> dict[str, Any]:
        """Live readiness probe: one uncached Context7 library search, policy-checked like a user command."""
        info = self.describe()
        if self.config.docs_provider == "none" or self.offline:
            return {**info, "ready": False, "reason": info["status"]}
        started = time.monotonic()
        try:
            self._authorize_user(Capability.NET_DOCS, self.docs_base_url, "research.check")
            api = Context7Client(
                self.client(),
                self.docs_base_url,
                await self._context7_key(),
                redact=self.rt.redactor.redact,
                now=self.rt.clock.now,
                retries=0,
            )
            results = await api.search("react", "hooks")
        except CoreError as exc:
            return {
                **info,
                "ready": False,
                "error_class": getattr(exc, "error_class", exc.code),
                "error": self.rt.redactor.redact(exc.message),
                "hint": exc.hint,
            }
        return {
            **info,
            "ready": True,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "results": len(results),
        }

    def _emit(
        self, kind: str, ctx: ToolContext | None, started: float, error: CoreError | None, **data: Any
    ) -> None:
        payload: dict[str, Any] = {
            "ok": error is None,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "via": "tool" if ctx is not None else "user",
            **{k: one_line(v, 300) if isinstance(v, str) else v for k, v in data.items()},
        }
        if error is not None:
            payload["error_class"] = getattr(error, "error_class", error.code)
            payload["error"] = one_line(error.message, 300)
        level = "info" if error is None else "warning"
        if ctx is not None:
            self.rt.events.emit(
                kind,
                project_id=ctx.project_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                attempt_id=ctx.attempt_id,
                level=level,
                data=payload,
            )
        else:
            project_id = self.rt.project.id if self.rt.project else None
            self.rt.events.emit(kind, project_id=project_id, actor="user", level=level, data=payload)
