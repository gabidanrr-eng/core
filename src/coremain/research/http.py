"""Bounded HTTP GET for research requests.

* URLs are validated before every hop: http(s) only, no embedded credentials, and link-local /
  cloud-metadata addresses are refused regardless of policy (they expose instance credentials).
* Redirects are followed manually so each hop is re-validated and re-authorized by the caller.
* Bodies are streamed and reading stops at ``max_bytes``.
* One bounded retry for connection failures and 500/502/503/504; everything else surfaces as a typed
  ``ResearchError`` (never a raw httpx exception).
"""

from __future__ import annotations

import asyncio
import codecs
import email.utils
import ipaddress
import json
import re
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from coremain.errors import PolicyDeniedError
from coremain.research.errors import ResearchError
from coremain.util.text import one_line

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
RETRY_STATUSES = frozenset({500, 502, 503, 504})
MAX_URL_LENGTH = 4096
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
TEXT_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-javascript",
        "application/ecmascript",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/x-sh",
        "application/sql",
        "application/graphql",
        "application/x-ndjson",
        "application/markdown",
    }
)
_METADATA_HOSTS = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "instance-data.ec2.internal",
    }
)
_METADATA_IPS = frozenset({ipaddress.ip_address("100.100.100.200"), ipaddress.ip_address("fd00:ec2::254")})
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:\-]+)""", re.IGNORECASE)


@dataclass
class HttpResponse:
    url: str
    status: int
    reason: str
    headers: httpx.Headers
    body: bytes
    truncated: bool
    redirects: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def charset(self) -> str | None:
        for param in self.headers.get("content-type", "").split(";")[1:]:
            name, _, value = param.partition("=")
            if name.strip().lower() == "charset" and value.strip():
                return value.strip().strip("\"'")
        return None

    def text(self) -> str:
        return decode_body(self)

    def json(self) -> Any:
        return json.loads(self.body.decode(self.charset or "utf-8", errors="replace"))


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse IP literals, including legacy IPv4 spellings (``2852039166``, ``0xa9fea9fe``) resolvers accept."""
    candidate = host.strip("[]")
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        pass
    if not candidate or not re.fullmatch(r"[0-9a-fA-FxX.]+", candidate):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(candidate))
    except OSError:
        return None


def is_local_host(host: str) -> bool:
    name = host.lower().strip("[]")
    if name == "localhost" or name.endswith(".localhost"):
        return True
    ip = literal_ip(name)
    return ip is not None and (ip.is_loopback or ip.is_unspecified)


def is_private_host(host: str) -> bool:
    """Loopback names or literal loopback/private/link-local/reserved addresses (no DNS lookup)."""
    if is_local_host(host):
        return True
    ip = literal_ip(host.lower())
    return ip is not None and (ip.is_private or ip.is_reserved or ip.is_link_local)


def guard_host(host: str) -> None:
    name = host.lower().strip("[]").rstrip(".")
    ip = literal_ip(name)
    mapped = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) else None
    blocked = name in _METADATA_HOSTS or (
        ip is not None and (ip.is_link_local or ip in _METADATA_IPS or bool(mapped and mapped.is_link_local))
    )
    if blocked:
        raise PolicyDeniedError(
            f"{host} is a link-local or cloud-metadata address; research requests never access it",
            details={"target": host},
        )


def validate_url(raw: str) -> str:
    """Return the normalized http(s) URL (fragment dropped) or raise a typed error."""
    url = raw.strip()
    if not url:
        raise ResearchError("empty URL", error_class="invalid_url")
    if len(url) > MAX_URL_LENGTH:
        raise ResearchError(f"URL is longer than {MAX_URL_LENGTH} characters", error_class="invalid_url")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port
    except ValueError as exc:
        raise ResearchError(f"invalid URL {one_line(url, 200)}: {exc}", error_class="invalid_url") from exc
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ResearchError(
            f"only http and https URLs can be fetched (got {scheme + ':' if scheme else 'no scheme'})",
            error_class="invalid_url",
        )
    if parts.username is not None or parts.password is not None:
        raise ResearchError(
            "URLs with embedded credentials are refused; research requests never send credentials",
            error_class="invalid_url",
        )
    if not host:
        raise ResearchError(f"URL has no host: {one_line(url, 200)}", error_class="invalid_url")
    guard_host(host)
    return urlunsplit((scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def parse_retry_after(headers: Mapping[str, str], now: float) -> int | None:
    value = (headers.get("retry-after") or "").strip()
    if value.isdigit():
        return int(value)
    if value:
        try:
            return max(0, int(email.utils.parsedate_to_datetime(value).timestamp() - now))
        except (TypeError, ValueError, IndexError):
            pass
    reset = (headers.get("ratelimit-reset") or "").strip()
    if reset.isdigit():
        n = int(reset)
        # Context7 documents RateLimit-Reset as a Unix timestamp; the IETF draft uses delta-seconds.
        return max(0, n - int(now)) if n > 1_000_000_000 else n
    return None


def decode_body(res: HttpResponse) -> str:
    body = res.body
    if body.startswith(codecs.BOM_UTF8):
        return body[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace")
    if body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return body.decode("utf-16", errors="replace")
    encoding = res.charset
    if encoding is None and (res.content_type in HTML_TYPES or not res.content_type):
        match = _META_CHARSET.search(body[:4096])
        encoding = match.group(1).decode("ascii", errors="ignore") if match else None
    try:
        codecs.lookup(encoding or "utf-8")
    except LookupError:
        encoding = None
    return body.decode(encoding or "utf-8", errors="replace")


def classify_content(content_type: str, body: bytes) -> Literal["html", "text", "binary"]:
    if content_type in HTML_TYPES:
        return "html"
    if content_type.startswith("text/") or content_type in TEXT_TYPES:
        return "text"
    if content_type.endswith(("+json", "+xml")):
        return "text"
    if content_type and content_type not in {"application/octet-stream", "binary/octet-stream"}:
        return "binary"
    head = body[:2048]
    if b"\x00" in head:
        return "binary"
    lowered = head.lstrip().lower()
    if lowered.startswith((b"<!doctype html", b"<html")):
        return "html"
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut at the sniff boundary is still text.
        if exc.start < len(head) - 4:
            return "binary"
    return "text"


def web_status_error(res: HttpResponse, now: float) -> ResearchError:
    code = res.status
    base = f"{res.url} returned HTTP {code}" + (f" {res.reason}" if res.reason else "")
    if code in {401, 403, 407}:
        return ResearchError(
            base + " (authentication required or access denied)",
            error_class="auth_failed",
            status=code,
            hint="research fetches never send credentials; use a public URL or a configured integration such as an MCP server",
        )
    if code in {404, 410}:
        return ResearchError(base, error_class="not_found", status=code)
    if code == 429:
        retry = parse_retry_after(res.headers, now)
        return ResearchError(
            base + " (rate limited" + (f"; retry after {retry}s)" if retry is not None else ")"),
            error_class="rate_limited",
            status=code,
            retry_after_s=retry,
        )
    if code >= 500:
        return ResearchError(base + " (server error)", error_class="upstream_error", status=code)
    if 300 <= code < 400:
        return ResearchError(
            base + " (redirect without a usable Location header)", error_class="http_error", status=code
        )
    return ResearchError(base, error_class="http_error", status=code)


async def _get_once(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, str] | None,
    headers: Mapping[str, str] | None,
    max_bytes: int,
) -> HttpResponse:
    host = host_of(url)
    try:
        async with client.stream("GET", url, params=params, headers=headers) as resp:
            chunks: list[bytes] = []
            total = 0
            truncated = False
            async for chunk in resp.aiter_bytes():
                if total + len(chunk) > max_bytes:
                    chunks.append(chunk[: max_bytes - total])
                    total = max_bytes
                    truncated = True
                    break
                chunks.append(chunk)
                total += len(chunk)
            return HttpResponse(
                url=str(resp.url),
                status=resp.status_code,
                reason=resp.reason_phrase,
                headers=resp.headers,
                body=b"".join(chunks),
                truncated=truncated,
            )
    except httpx.TimeoutException as exc:
        raise ResearchError(
            f"timed out contacting {host} ({type(exc).__name__})", error_class="timeout"
        ) from exc
    except httpx.UnsupportedProtocol as exc:
        raise ResearchError(f"unsupported URL {one_line(url, 200)}", error_class="invalid_url") from exc
    except httpx.DecodingError as exc:
        raise ResearchError(
            f"could not decode the response from {host}: {one_line(str(exc), 200)}",
            error_class="invalid_response",
        ) from exc
    except httpx.RequestError as exc:
        raise ResearchError(
            f"network error contacting {host}: {type(exc).__name__}: {one_line(str(exc), 200)}",
            error_class="network_error",
        ) from exc
    except httpx.InvalidURL as exc:
        raise ResearchError(f"invalid URL {one_line(url, 200)}: {exc}", error_class="invalid_url") from exc


async def bounded_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    max_bytes: int,
    max_redirects: int = 0,
    authorize_hop: Callable[[str], None] | None = None,
    retries: int = 1,
    backoff_s: float = 0.5,
) -> HttpResponse:
    """GET ``url``; 3xx responses are returned as-is when ``max_redirects`` is 0."""
    started = time.monotonic()
    redirects: list[str] = []
    current = url
    while True:
        resp = await _with_retry(
            client,
            current,
            params=None if redirects else params,
            headers=headers,
            max_bytes=max_bytes,
            retries=retries,
            backoff_s=backoff_s,
        )
        location = resp.headers.get("location")
        if max_redirects > 0 and resp.status in REDIRECT_STATUSES and location:
            if len(redirects) >= max_redirects:
                raise ResearchError(
                    f"more than {max_redirects} redirects starting at {url}", error_class="redirect_limit"
                )
            target = validate_url(urljoin(resp.url, location.strip()))
            if authorize_hop is not None:
                authorize_hop(target)
            redirects.append(target)
            current = target
            continue
        resp.redirects = redirects
        resp.elapsed_ms = int((time.monotonic() - started) * 1000)
        return resp


async def _with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, str] | None,
    headers: Mapping[str, str] | None,
    max_bytes: int,
    retries: int,
    backoff_s: float,
) -> HttpResponse:
    attempt = 0
    while True:
        try:
            resp = await _get_once(client, url, params=params, headers=headers, max_bytes=max_bytes)
        except ResearchError as exc:
            if exc.error_class != "network_error" or attempt >= retries:
                raise
        else:
            if resp.status not in RETRY_STATUSES or attempt >= retries:
                return resp
        attempt += 1
        await asyncio.sleep(backoff_s * attempt)
