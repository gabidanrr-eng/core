"""Origin allowlisting, file confinement and target normalization for browser automation.

Pattern syntax for ``browser.allowed_origins``:

* ``*`` – any http(s)/ws(s) origin and any file URL (file confinement still applies).
* ``scheme://host[:port][/path]`` – ``scheme`` is ``http``, ``https``, ``ws``, ``wss`` or ``*``;
  ``host`` may use ``*`` wildcards (``*.example.com`` also matches ``example.com``); a missing
  port means the scheme's default port and ``*`` means any port; an optional path is an
  fnmatch pattern. ``http`` patterns also cover ``ws`` and ``https`` patterns cover ``wss``.
* ``host[:port]`` – shorthand for any network scheme.
* ``file://*`` or ``file:///some/dir/*`` – local files (always additionally confined to the
  workspace by :class:`FileScope`).

Invalid patterns never allow anything; they are reported by ``BrowserManager.check``.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import url2pathname

from coremain.config.schema import CoreConfig
from coremain.errors import UsageError
from coremain.security.paths import is_within, resolve_within, sensitive_reason, to_relative
from coremain.security.policy import domain_matches

DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}
NETWORK_SCHEMES = frozenset(DEFAULT_PORTS)
_PATTERN_SCHEMES = frozenset({"*", "http", "https", "ws", "wss"})
_EQUIVALENT = {"ws": "http", "wss": "https"}
_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
_PATTERN_RE = re.compile(r"^([A-Za-z*][A-Za-z0-9+.*-]*)://(.*)$")
ALLOW_HINT = "add the origin to browser.allowed_origins in your user configuration if it should be reachable"


class TargetError(UsageError):
    code = "invalid_browser_target"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str
    target: str


def is_loopback_host(host: str) -> bool:
    host = host.strip("[]").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def origin_of(scheme: str, host: str, port: int | None) -> str:
    shown = f"[{host}]" if ":" in host else host
    if port is None or port == DEFAULT_PORTS.get(scheme):
        return f"{scheme}://{shown}"
    return f"{scheme}://{shown}:{port}"


def file_url_to_path(url: str) -> Path | None:
    """Local absolute path of a ``file://`` URL, or ``None`` for remote/malformed file URLs."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() != "file" or parts.netloc.lower() not in ("", "localhost"):
        return None
    raw = url2pathname(parts.path)
    if not raw or "\x00" in raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else None


def _host_matches(pattern: str, host: str) -> bool:
    if pattern == "*":
        return True
    if pattern.startswith("*.") and host == pattern[2:]:
        return True
    return fnmatch.fnmatchcase(host, pattern)


@dataclass(frozen=True)
class OriginPattern:
    raw: str
    scheme: str
    host: str | None = None
    port: str | None = None
    path: str | None = None

    @classmethod
    def parse(cls, raw: str) -> OriginPattern:
        text = raw.strip()
        if not text:
            raise ValueError("empty pattern")
        if text == "*":
            return cls(text, "*", "*", "*", None)
        m = _PATTERN_RE.match(text)
        scheme, rest = (m.group(1).lower(), m.group(2)) if m else ("*", text)
        if scheme == "file":
            if rest in ("", "*", "/*"):
                return cls(text, "file")
            if not rest.startswith("/"):
                host_part, _, tail = rest.partition("/")
                if host_part.lower() not in ("localhost", "*"):
                    raise ValueError("file patterns must not name a remote host")
                rest = "/" + tail
            return cls(text, "file", path=rest)
        if scheme not in _PATTERN_SCHEMES:
            raise ValueError(f"unsupported scheme '{scheme}'")
        hostport, slash, tail = rest.partition("/")
        path = None if not slash or tail in ("", "*") else "/" + tail
        port: str | None
        if hostport.startswith("["):
            end = hostport.find("]")
            if end < 0:
                raise ValueError("unterminated IPv6 literal")
            host, after = hostport[1:end], hostport[end + 1 :]
            if after and not after.startswith(":"):
                raise ValueError("unexpected text after IPv6 literal")
            port = after[1:] if after else None
        elif hostport.count(":") == 1:
            host, port = hostport.split(":")
        elif ":" not in hostport:
            host, port = hostport, None
        else:
            raise ValueError("IPv6 hosts must be written in brackets, e.g. http://[::1]:*")
        if not host:
            raise ValueError("missing host")
        if port is not None and port != "*" and not port.isdigit():
            raise ValueError(f"invalid port '{port}'")
        return cls(text, scheme, host.lower().rstrip("."), port, path)

    def matches_network(self, scheme: str, host: str, port: int, path: str | None) -> bool:
        if self.scheme == "file" or self.host is None:
            return False
        if self.scheme != "*" and self.scheme not in (scheme, _EQUIVALENT.get(scheme)):
            return False
        if not _host_matches(self.host, host):
            return False
        if self.port is None:
            if port != DEFAULT_PORTS[scheme]:
                return False
        elif self.port != "*" and int(self.port) != port:
            return False
        # ``path is None`` means the caller cannot see the path (a CONNECT tunnel).
        return self.path is None or path is None or fnmatch.fnmatchcase(path, self.path)

    def matches_file(self, path: str) -> bool:
        if self.raw == "*":
            return True
        if self.scheme != "file":
            return False
        return self.path is None or fnmatch.fnmatchcase(path, self.path)


class OriginPolicy:
    """The single origin verdict used by tools, the route handler, the egress proxy and checks."""

    def __init__(self, patterns: Iterable[str], *, offline: bool = False, deny_domains: Sequence[str] = ()):
        self.patterns: list[OriginPattern] = []
        self.invalid: list[str] = []
        for raw in patterns:
            try:
                self.patterns.append(OriginPattern.parse(raw))
            except ValueError as exc:
                self.invalid.append(f"{raw!r}: {exc}")
        self.offline = offline
        self.deny_domains = list(deny_domains)

    @classmethod
    def from_config(cls, config: CoreConfig) -> OriginPolicy:
        return cls(
            config.browser.allowed_origins,
            offline=config.permissions.network.offline,
            deny_domains=config.permissions.network.deny_domains,
        )

    def check_network(self, scheme: str, host: str, port: int, path: str | None = "/") -> Verdict:
        scheme = scheme.lower()
        host = host.strip("[]").lower().rstrip(".")
        origin = origin_of(scheme, host, port)
        if scheme not in NETWORK_SCHEMES or not host:
            return Verdict(False, "not a network origin", origin)
        if self.offline and not is_loopback_host(host):
            return Verdict(
                False,
                "offline mode (permissions.network.offline): only loopback origins and workspace files are reachable",
                origin,
            )
        if self.deny_domains and domain_matches(host, self.deny_domains):
            return Verdict(False, f"{host} is listed in permissions.network.deny_domains", origin)
        for pattern in self.patterns:
            if pattern.matches_network(scheme, host, port, path):
                return Verdict(True, f"allowed by browser.allowed_origins entry '{pattern.raw}'", origin)
        return Verdict(False, f"{origin} is not in browser.allowed_origins", origin)

    def check_connect(self, host: str, port: int) -> Verdict:
        """Verdict for a CONNECT tunnel, which may carry https, wss or a plain ws:// upgrade."""
        verdicts = [self.check_network(s, host, port, None) for s in ("https", "wss", "http", "ws")]
        return next((v for v in verdicts if v.allowed), verdicts[0])

    def check_url(self, url: str) -> Verdict:
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            return Verdict(False, "malformed URL", url[:200])
        scheme = parts.scheme.lower()
        if scheme in NETWORK_SCHEMES:
            host = (parts.hostname or "").lower()
            if not host:
                return Verdict(False, "URL has no host", url[:200])
            return self.check_network(scheme, host, port or DEFAULT_PORTS[scheme], parts.path or "/")
        if scheme == "file":
            path = file_url_to_path(url)
            if path is None:
                return Verdict(False, "only local file:// URLs are allowed", url[:200])
            if any(p.matches_file(str(path)) for p in self.patterns):
                return Verdict(True, "file URLs are allowed by browser.allowed_origins", str(path))
            return Verdict(False, "file URLs are not in browser.allowed_origins", str(path))
        if scheme == "about":
            ok = url in ("about:blank", "about:srcdoc")
            return Verdict(ok, "local page" if ok else "only about:blank is allowed", url[:200])
        if scheme in ("data", "blob"):
            # Inline content never reaches the network or disk by itself; what it loads is checked.
            return Verdict(True, "inline content", scheme)
        return Verdict(False, f"URL scheme '{scheme}:' is not allowed", url[:200])


@dataclass(frozen=True)
class FileScope:
    """Directories whose files a browser session may load; sensitive files are always refused."""

    roots: tuple[Path, ...]
    extra_sensitive: tuple[str, ...] = ()

    @classmethod
    def of(cls, *roots: Path, extra_sensitive: Iterable[str] = ()) -> FileScope:
        return cls(tuple(r.resolve() for r in roots), tuple(extra_sensitive))

    def check(self, url: str) -> Verdict:
        path = file_url_to_path(url)
        if path is None:
            return Verdict(False, "not a local file URL", url[:200])
        resolved = path.resolve(strict=False)
        for root in self.roots:
            if not is_within(root, resolved):
                continue
            rel = to_relative(root, resolved)
            if Path(rel).parts[:1] == (".git",):
                return Verdict(False, "the .git directory is not readable through the browser", str(resolved))
            reason = sensitive_reason(rel, self.extra_sensitive)
            if reason:
                return Verdict(False, f"sensitive file ({reason})", str(resolved))
            return Verdict(True, "inside the workspace", str(resolved))
        return Verdict(False, "file is outside the workspace", str(resolved))


def check_target(origins: OriginPolicy, files: FileScope, url: str) -> Verdict:
    """Origin verdict plus, for file URLs, workspace confinement: the rule for every request."""
    verdict = origins.check_url(url)
    if verdict.allowed and url[:5].lower() == "file:":
        return files.check(url)
    return verdict


def policy_target(url: str) -> str:
    """Policy-engine target for a browser URL: the origin for network URLs, the file URL for files.

    Origins (not full URLs) keep approval grants meaningful: approving ``https://example.com``
    for a task covers every page and interaction on that origin, mirroring the same-origin model.
    Browser-internal pages (``about:``, ``chrome-error:``, ``data:``, ``blob:``) map to the empty
    (local) target: they reach neither the network nor the filesystem by themselves.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return url[:500]
    scheme = parts.scheme.lower()
    if scheme in NETWORK_SCHEMES:
        return origin_of(scheme, (parts.hostname or "").lower(), port)
    if scheme == "file":
        return urlunsplit(("file", "", parts.path, "", ""))
    return ""


def _split_suffix(text: str, base: Path) -> tuple[str, str]:
    cut = min((i for i in (text.find("?"), text.find("#")) if i >= 0), default=-1)
    if cut <= 0 or (base / text).exists():
        return text, ""
    return text[:cut], text[cut:]


def resolve_target(raw: str, base: Path, *, confine: bool = True) -> str:
    """Normalize a browser target into a URL.

    ``http(s)://`` and ``file://`` URLs are validated; anything without a scheme is a path
    relative to ``base`` (the task workspace) turned into a ``file://`` URL. With ``confine``
    (tools and checks) paths and file URLs must stay inside ``base``.
    """
    text = raw.strip()
    if not text:
        raise TargetError("empty URL")
    if text == "about:blank":
        return text
    m = _SCHEME_RE.match(text)
    if m and len(m.group(1)) > 1:
        scheme = m.group(1).lower()
        if scheme in ("http", "https"):
            try:
                parts = urlsplit(text)
                _ = parts.port
            except ValueError as exc:
                raise TargetError(f"malformed URL '{text[:200]}': {exc}") from exc
            if not parts.hostname:
                raise TargetError(f"URL '{text[:200]}' has no host")
            return text
        if scheme == "file":
            path = file_url_to_path(text)
            if path is None:
                raise TargetError("only local file:// URLs with an absolute path are supported")
            if confine:
                resolve_within(base, path)
            if not path.exists():
                raise TargetError(f"{path} does not exist")
            return text
        hint = "use an http://, https:// or file:// URL, or a path relative to the workspace"
        rest = text[len(scheme) + 1 :]
        if rest[:1].isdigit():
            hint = f"did you mean http://{text}?"
        raise TargetError(f"unsupported URL scheme '{scheme}:'", hint=hint)
    path_part, suffix = _split_suffix(text, base)
    if confine:
        resolved = resolve_within(base, path_part)
    else:
        candidate = Path(os.path.expanduser(path_part))
        resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not resolved.exists():
        raise TargetError(
            f"'{path_part}' does not exist", hint="paths are resolved relative to the workspace root"
        )
    return resolved.as_uri() + suffix
