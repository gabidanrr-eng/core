"""Per-session loopback egress proxy enforcing the browser origin policy on every connection.

Playwright route handlers only see the first URL of a redirect chain and never see WebSockets,
so they cannot stop a page from leaving the allowed origins. Chromium, however, sends every
http(s)/ws(s) connection of a context through that context's proxy (Playwright also forces
loopback traffic through it), so this proxy is the network boundary:

* plain HTTP requests arrive in absolute form and are checked per URL, then forwarded as one
  request per connection (``Connection: close`` both ways, so connections are never reused
  for a different host);
* CONNECT tunnels (https, wss and ws) are checked by host:port and relayed byte-for-byte, so
  TLS stays end-to-end;
* loopback host names are only ever connected to loopback addresses, so DNS cannot turn an
  allowed local name into network egress.

Refusals are explicit responses (403 with ``X-Core-Main-Blocked``, 502/504 with
``X-Core-Main-Proxy-Error``) and are reported through ``on_record``.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import ipaddress
import socket
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from coremain.browser.origins import OriginPolicy, is_loopback_host

HEAD_LIMIT = 64 * 1024
HEAD_TIMEOUT_S = 30.0
CONNECT_TIMEOUT_S = 15.0
TUNNEL_DRAIN_S = 30.0
CHUNK = 64 * 1024
BLOCKED_HEADER = "X-Core-Main-Blocked"
ERROR_HEADER = "X-Core-Main-Proxy-Error"
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "proxy-authorization",
        "proxy-authenticate",
        "te",
        "trailer",
        "upgrade",
    }
)
_RESPONSE_HOP = (b"connection", b"keep-alive", b"proxy-connection")


@dataclass(frozen=True)
class EgressRecord:
    kind: str  # "blocked" | "upstream_error"
    target: str
    reason: str
    method: str
    url: str | None
    ts: float

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


class _UpstreamError(Exception):
    def __init__(self, kind: str, reason: str, status: int):
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.status = status


def _split_authority(authority: str) -> tuple[str, int] | None:
    if authority.startswith("["):
        end = authority.find("]")
        if end < 0 or authority[end + 1 : end + 2] != ":":
            return None
        host, port = authority[1:end], authority[end + 2 :]
    else:
        host, sep, port = authority.rpartition(":")
        if not sep:
            return None
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        return None
    return host.lower(), int(port)


def _filter_request_headers(blob: bytes) -> bytes:
    parsed: list[tuple[str, bytes]] = []
    listed: set[str] = set()
    for raw in blob.split(b"\r\n") if blob else []:
        name, sep, value = raw.partition(b":")
        if not sep or not name.strip():
            continue
        lname = name.strip().lower().decode("latin-1")
        parsed.append((lname, raw))
        if lname == "connection":
            listed |= {t.strip().lower() for t in value.decode("latin-1").split(",") if t.strip()}
    return b"".join(
        raw + b"\r\n" for lname, raw in parsed if lname not in _HOP_BY_HOP and lname not in listed
    )


def _describe(exc: BaseException) -> tuple[str, str, int]:
    if isinstance(exc, TimeoutError):
        return "timeout", "connection timed out", 504
    if isinstance(exc, socket.gaierror):
        return "name_not_resolved", "host name could not be resolved", 502
    if isinstance(exc, ConnectionRefusedError):
        return "connection_refused", "connection refused (is the server running?)", 502
    if isinstance(exc, OSError):
        return "connect_failed", exc.strerror or str(exc) or type(exc).__name__, 502
    return "connect_failed", str(exc) or type(exc).__name__, 502


async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await src.read(CHUNK)
            if not chunk:
                break
            dst.write(chunk)
            await dst.drain()
    except (ConnectionError, OSError):
        return
    if dst.can_write_eof():
        with contextlib.suppress(OSError, RuntimeError):
            dst.write_eof()


async def _pipe_response(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    """Relay an HTTP response, forcing ``Connection: close`` on the final (non-1xx) head."""
    try:
        while True:
            head = await src.readuntil(b"\r\n\r\n")
            status_line, _, rest = head[:-4].partition(b"\r\n")
            fields = status_line.split(b" ", 2)
            code = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else 0
            if 100 <= code < 200 and code != 101:
                dst.write(head)
                await dst.drain()
                continue
            kept = [
                h
                for h in rest.split(b"\r\n")
                if h and h.split(b":", 1)[0].strip().lower() not in _RESPONSE_HOP
            ]
            dst.write(
                status_line + b"\r\n" + b"".join(h + b"\r\n" for h in kept) + b"Connection: close\r\n\r\n"
            )
            await dst.drain()
            break
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, OSError):
        return
    await _pipe(src, dst)


class EgressProxy:
    def __init__(
        self,
        policy: OriginPolicy,
        *,
        on_record: Callable[[EgressRecord], None] | None = None,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
    ):
        self.policy = policy
        self.on_record = on_record
        self.connect_timeout_s = connect_timeout_s
        self.port: int | None = None
        self.records: deque[EgressRecord] = deque(maxlen=200)
        self.stats = {"requests": 0, "tunnels": 0, "blocked": 0, "errors": 0}
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def url(self) -> str:
        if self.port is None:
            raise RuntimeError("egress proxy is not running")
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return self._server is not None

    async def start(self) -> str:
        if self._server is None:
            self._server = await asyncio.start_server(
                self._handle, host="127.0.0.1", port=0, limit=HEAD_LIMIT
            )
            self.port = self._server.sockets[0].getsockname()[1]
        return self.url

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(server.wait_closed(), timeout=5)

    # ------------------------------------------------------------------ helpers
    def _record(self, kind: str, target: str, reason: str, method: str, url: str | None) -> None:
        record = EgressRecord(kind, target, reason, method, url[:500] if url else None, time.time())
        self.records.append(record)
        self.stats["blocked" if kind == "blocked" else "errors"] += 1
        if self.on_record is not None:
            with contextlib.suppress(Exception):
                self.on_record(record)

    async def _respond(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        phrase: str,
        message: str,
        *,
        header: tuple[str, str] | None = None,
        body: bool = True,
    ) -> None:
        payload = (
            (
                "<!doctype html><html><head><title>Blocked by Core Main</title></head><body>"
                f"<h1>Core Main browser proxy: {status} {html.escape(phrase)}</h1>"
                f"<p>{html.escape(message)}</p></body></html>"
            ).encode()
            if body
            else b""
        )
        lines = [
            f"HTTP/1.1 {status} {phrase}",
            "Content-Type: text/html; charset=utf-8",
            f"Content-Length: {len(payload)}",
            "Cache-Control: no-store",
            "Connection: close",
        ]
        if header is not None:
            lines.append(f"{header[0]}: {header[1]}")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + payload)
        with contextlib.suppress(ConnectionError, OSError):
            await writer.drain()

    async def _addresses(self, host: str, port: int) -> list[str]:
        if not is_loopback_host(host):
            return [host]
        try:
            return [str(ipaddress.ip_address(host.strip("[]")))]
        except ValueError:
            pass
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError:
            infos = []
        found: list[str] = []
        for info in infos:
            addr = str(info[4][0])
            with contextlib.suppress(ValueError):
                if ipaddress.ip_address(addr.split("%")[0]).is_loopback and addr not in found:
                    found.append(addr)
        # RFC 6761: *.localhost always means loopback, whatever the resolver says.
        return found or ["127.0.0.1", "::1"]

    async def _open(self, host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        last: BaseException | None = None
        for addr in await self._addresses(host, port):
            try:
                return await asyncio.wait_for(
                    asyncio.open_connection(addr, port, limit=HEAD_LIMIT), timeout=self.connect_timeout_s
                )
            except (OSError, TimeoutError) as exc:
                last = exc
        kind, reason, status = _describe(last or OSError("no address"))
        raise _UpstreamError(kind, reason, status)

    # ------------------------------------------------------------------ serving
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            await self._serve(reader, writer)
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            if task is not None:
                self._tasks.discard(task)
            writer.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=HEAD_TIMEOUT_S)
        except asyncio.LimitOverrunError:
            await self._respond(writer, 431, "Request Header Fields Too Large", "request head too large")
            return
        except (asyncio.IncompleteReadError, TimeoutError):
            return
        line, _, header_blob = head[:-4].partition(b"\r\n")
        try:
            method, target, version = line.decode("latin-1").split(" ")
        except ValueError:
            await self._respond(writer, 400, "Bad Request", "malformed request line")
            return
        if not version.startswith("HTTP/1."):
            await self._respond(writer, 400, "Bad Request", "only HTTP/1.x is supported")
            return
        if method.upper() == "CONNECT":
            await self._tunnel(target, reader, writer)
        else:
            await self._forward(method, target, version, header_blob, reader, writer)

    async def _tunnel(
        self, authority: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        parsed = _split_authority(authority)
        if parsed is None:
            await self._respond(writer, 400, "Bad Request", "CONNECT requires host:port", body=False)
            return
        host, port = parsed
        verdict = self.policy.check_connect(host, port)
        if not verdict.allowed:
            self._record("blocked", f"{host}:{port}", verdict.reason, "CONNECT", None)
            await self._respond(
                writer, 403, "Forbidden", verdict.reason, header=(BLOCKED_HEADER, "1"), body=False
            )
            return
        try:
            up_r, up_w = await self._open(host, port)
        except _UpstreamError as exc:
            self._record("upstream_error", f"{host}:{port}", exc.reason, "CONNECT", None)
            await self._respond(
                writer, exc.status, "Bad Gateway", exc.reason, header=(ERROR_HEADER, exc.kind), body=False
            )
            return
        self.stats["tunnels"] += 1
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        up = asyncio.create_task(_pipe(reader, up_w))
        down = asyncio.create_task(_pipe(up_r, writer))
        try:
            _, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
            if pending:
                await asyncio.wait(pending, timeout=TUNNEL_DRAIN_S)
        finally:
            await self._finish(up, down, up_w)

    async def _forward(
        self,
        method: str,
        target: str,
        version: str,
        header_blob: bytes,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            parts = urlsplit(target)
            port = parts.port or 80
        except ValueError:
            await self._respond(writer, 400, "Bad Request", "malformed request target")
            return
        host = (parts.hostname or "").lower()
        if parts.scheme.lower() != "http" or not host:
            await self._respond(writer, 400, "Bad Request", "proxy requests must use an absolute http:// URL")
            return
        path = parts.path or "/"
        verdict = self.policy.check_network("http", host, port, path)
        if not verdict.allowed:
            self._record("blocked", verdict.target, verdict.reason, method, target)
            await self._respond(
                writer, 403, "Forbidden", f"{verdict.target}: {verdict.reason}", header=(BLOCKED_HEADER, "1")
            )
            return
        try:
            up_r, up_w = await self._open(host, port)
        except _UpstreamError as exc:
            self._record("upstream_error", verdict.target, exc.reason, method, target)
            phrase = "Gateway Timeout" if exc.status == 504 else "Bad Gateway"
            await self._respond(
                writer, exc.status, phrase, f"{verdict.target}: {exc.reason}", header=(ERROR_HEADER, exc.kind)
            )
            return
        self.stats["requests"] += 1
        request_target = path + (f"?{parts.query}" if parts.query else "")
        up_w.write(
            f"{method} {request_target} {version}\r\n".encode("latin-1")
            + _filter_request_headers(header_blob)
            + b"Connection: close\r\n\r\n"
        )
        up = asyncio.create_task(_pipe(reader, up_w))
        down = asyncio.create_task(_pipe_response(up_r, writer))
        try:
            # The exchange is over once the response has been relayed; a client that half-closes
            # after sending its request still gets a bounded wait for the response.
            done, _ = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
            if down not in done:
                await asyncio.wait({down}, timeout=TUNNEL_DRAIN_S)
        finally:
            await self._finish(up, down, up_w)

    @staticmethod
    async def _finish(up: asyncio.Task[None], down: asyncio.Task[None], up_w: asyncio.StreamWriter) -> None:
        for task in (up, down):
            task.cancel()
        await asyncio.gather(up, down, return_exceptions=True)
        up_w.close()
