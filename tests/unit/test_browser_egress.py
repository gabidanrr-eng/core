"""The browser egress proxy enforces the origin policy on every connection (no browser needed)."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import socket
from collections.abc import AsyncIterator

import pytest

from coremain.browser.egress import BLOCKED_HEADER, ERROR_HEADER, EgressProxy, EgressRecord
from coremain.browser.origins import OriginPolicy


class Upstream:
    """A tiny HTTP server (or echo server) that records what reaches it."""

    def __init__(self, *, echo: bool = False):
        self.echo = echo
        self.heads: list[str] = []
        self.connections = 0
        self.server: asyncio.Server | None = None
        self.port = 0

    async def start(self) -> Upstream:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            if self.echo:
                while data := await reader.read(4096):
                    writer.write(data)
                    await writer.drain()
                return
            head = await reader.readuntil(b"\r\n\r\n")
            self.heads.append(head.decode("latin-1"))
            body = b"hello from upstream"
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nKeep-Alive: timeout=5\r\n"
                b"Connection: keep-alive\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)
            )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.asynccontextmanager
async def proxy_for(
    *patterns: str, offline: bool = False
) -> AsyncIterator[tuple[EgressProxy, list[EgressRecord]]]:
    records: list[EgressRecord] = []
    proxy = EgressProxy(
        OriginPolicy(patterns, offline=offline), on_record=records.append, connect_timeout_s=5
    )
    await proxy.start()
    try:
        yield proxy, records
    finally:
        await proxy.close()


async def exchange(proxy: EgressProxy, data: bytes) -> bytes:
    assert proxy.port is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
    writer.write(data)
    await writer.drain()
    try:
        return await asyncio.wait_for(reader.read(), timeout=10)
    finally:
        writer.close()


def head_of(response: bytes) -> str:
    return response.split(b"\r\n\r\n", 1)[0].decode("latin-1")


async def test_allowed_http_request_is_forwarded_in_origin_form_without_hop_headers() -> None:
    up = await Upstream().start()
    try:
        async with proxy_for(f"http://127.0.0.1:{up.port}") as (proxy, records):
            response = await exchange(
                proxy,
                (
                    f"GET http://127.0.0.1:{up.port}/hello?x=1 HTTP/1.1\r\nHost: 127.0.0.1:{up.port}\r\n"
                    "Proxy-Connection: keep-alive\r\nConnection: keep-alive, X-Hop\r\nX-Hop: 1\r\n"
                    "Proxy-Authorization: Basic c2VjcmV0\r\nX-Keep: yes\r\n\r\n"
                ).encode(),
            )
    finally:
        await up.close()
    assert response.startswith(b"HTTP/1.1 200 OK")
    assert response.endswith(b"hello from upstream")
    head = head_of(response)
    assert "Connection: close" in head
    assert "keep-alive" not in head.lower()
    assert len(up.heads) == 1
    sent = up.heads[0]
    assert sent.startswith("GET /hello?x=1 HTTP/1.1\r\n")
    assert "X-Keep: yes" in sent and "Connection: close" in sent
    for hop in ("Proxy-Connection", "X-Hop", "Proxy-Authorization", "keep-alive"):
        assert hop not in sent
    assert records == []
    assert proxy.stats["requests"] == 1


async def test_disallowed_origin_gets_an_explicit_block_page_and_never_reaches_upstream() -> None:
    up = await Upstream().start()
    try:
        async with proxy_for("http://127.0.0.1:1") as (proxy, records):
            response = await exchange(
                proxy, f"GET http://127.0.0.1:{up.port}/secret HTTP/1.1\r\nHost: x\r\n\r\n".encode()
            )
    finally:
        await up.close()
    assert response.startswith(b"HTTP/1.1 403 Forbidden")
    assert f"{BLOCKED_HEADER}: 1" in head_of(response)
    assert b"is not in browser.allowed_origins" in response
    assert up.connections == 0
    assert [(r.kind, r.target, r.method) for r in records] == [
        ("blocked", f"http://127.0.0.1:{up.port}", "GET")
    ]


async def test_connect_tunnels_are_checked_by_host_and_port() -> None:
    echo = await Upstream(echo=True).start()
    try:
        async with proxy_for(f"https://127.0.0.1:{echo.port}") as (proxy, records):
            assert proxy.port is not None
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
            writer.write(
                f"CONNECT 127.0.0.1:{echo.port} HTTP/1.1\r\nHost: 127.0.0.1:{echo.port}\r\n\r\n".encode()
            )
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            assert head.startswith(b"HTTP/1.1 200 Connection Established")
            writer.write(b"opaque tls bytes")
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(16), 5) == b"opaque tls bytes"
            writer.close()

            blocked = await exchange(proxy, b"CONNECT 127.0.0.1:9 HTTP/1.1\r\nHost: 127.0.0.1:9\r\n\r\n")
            assert blocked.startswith(b"HTTP/1.1 403 Forbidden")
            assert records[-1].kind == "blocked" and records[-1].target == "127.0.0.1:9"
            assert proxy.stats["tunnels"] == 1
    finally:
        await echo.close()


async def test_unreachable_upstream_is_reported_not_hidden() -> None:
    port = free_port()
    async with proxy_for(f"http://127.0.0.1:{port}") as (proxy, records):
        response = await exchange(proxy, f"GET http://127.0.0.1:{port}/ HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        tunnel = await exchange(proxy, f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
    assert response.startswith(b"HTTP/1.1 502 Bad Gateway")
    assert f"{ERROR_HEADER}: connection_refused" in head_of(response)
    assert b"connection refused" in response
    assert tunnel.startswith(b"HTTP/1.1 502")
    assert [r.kind for r in records] == ["upstream_error", "upstream_error"]


async def test_offline_mode_refuses_non_loopback_before_resolving() -> None:
    async with proxy_for("*", offline=True) as (proxy, records):
        response = await exchange(proxy, b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
        plain = await exchange(proxy, b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
    assert response.startswith(b"HTTP/1.1 403")
    assert plain.startswith(b"HTTP/1.1 403") and b"offline mode" in plain
    assert all("offline mode" in r.reason for r in records) and len(records) == 2


@pytest.mark.parametrize(
    ("request_bytes", "status"),
    [
        (b"NONSENSE\r\n\r\n", b"400"),
        (b"GET /relative HTTP/1.1\r\nHost: x\r\n\r\n", b"400"),
        (b"GET https://example.com/ HTTP/1.1\r\n\r\n", b"400"),
        (b"CONNECT nohostport HTTP/1.1\r\n\r\n", b"400"),
        (b"GET http://x/ HTTP/2\r\n\r\n", b"400"),
        (b"GET http://x/ HTTP/1.1\r\nX-Big: " + b"a" * 70_000 + b"\r\n\r\n", b"431"),
    ],
)
async def test_malformed_requests_are_rejected(request_bytes: bytes, status: bytes) -> None:
    async with proxy_for("*") as (proxy, _):
        response = await exchange(proxy, request_bytes)
    assert response.split(b" ")[1] == status


async def test_close_terminates_open_tunnels() -> None:
    echo = await Upstream(echo=True).start()
    try:
        proxy = EgressProxy(OriginPolicy([f"http://127.0.0.1:{echo.port}"]))
        await proxy.start()
        assert proxy.port is not None
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
        writer.write(f"CONNECT 127.0.0.1:{echo.port} HTTP/1.1\r\n\r\n".encode())
        await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        await asyncio.wait_for(proxy.close(), 10)
        assert await asyncio.wait_for(reader.read(), 5) == b""
        writer.close()
        assert not proxy.running
    finally:
        await echo.close()


async def test_loopback_names_are_pinned_to_loopback_addresses() -> None:
    proxy = EgressProxy(OriginPolicy(["*"]))
    for host in ("localhost", "app.localhost", "127.0.0.1", "::1"):
        addresses = await proxy._addresses(host, 80)
        assert addresses and all(ipaddress.ip_address(a.split("%")[0]).is_loopback for a in addresses), host
    assert await proxy._addresses("example.com", 443) == ["example.com"]
