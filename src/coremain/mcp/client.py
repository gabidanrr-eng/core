"""MCP client.

Era detection follows the specification's guidance: probe with ``server/discover`` (modern,
stateless, every request carries ``params._meta``); a DiscoverResult or a recognised modern
error means modern, anything else falls back to the legacy ``initialize`` handshake. The
fallback is never keyed on a single error code. Server-initiated requests are answered with
"method not found" (except legacy ``ping``) and ``input_required`` results are surfaced as an
explicit unsupported-interaction error rather than being guessed at.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from coremain.errors import CoreError, ExitCode
from coremain.providers.sse import iter_sse
from coremain.version import __version__

MODERN_VERSION = "2026-07-28"
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
CLIENT_INFO = {"name": "core-main", "version": __version__}
MODERN_ERRORS = {-32022, -32021, -32020}
META = "io.modelcontextprotocol"


class MCPError(CoreError):
    code = "mcp_error"

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        rpc_code: int | None = None,
        data: Any = None,
        hint: str | None = None,
    ):
        super().__init__(message, hint=hint, details={"error_class": error_class, "rpc_code": rpc_code})
        self.error_class = error_class
        self.rpc_code = rpc_code
        self.data = data
        self.exit_code = (
            ExitCode.BLOCKED
            if error_class in {"unavailable", "auth_required", "unsupported"}
            else ExitCode.FAILURE
        )


def modern_meta() -> dict[str, Any]:
    return {
        f"{META}/protocolVersion": MODERN_VERSION,
        f"{META}/clientCapabilities": {},
        f"{META}/clientInfo": CLIENT_INFO,
    }


def _rpc_error(err: dict[str, Any]) -> MCPError:
    code = err.get("code")
    message = str(err.get("message", "error"))
    klass = {
        -32601: "method_not_found",
        -32602: "invalid_params",
        -32600: "invalid_request",
        -32700: "parse_error",
        -32022: "unsupported_version",
        -32021: "missing_capability",
        -32020: "header_mismatch",
        -32002: "not_found",
    }.get(code, "rpc_error")
    return MCPError(f"server error {code}: {message}", error_class=klass, rpc_code=code, data=err.get("data"))


class Transport:
    async def start(self) -> None: ...

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def notify(
        self, method: str, params: dict[str, Any] | None = None, *, headers: dict[str, str] | None = None
    ) -> None: ...

    async def close(self) -> None: ...

    def diagnostics(self) -> str:
        return ""


class StdioTransport(Transport):
    def __init__(
        self,
        command: list[str],
        env: dict[str, str],
        cwd: str | None = None,
        *,
        startup_timeout: float = 30.0,
    ):
        self.command = command
        self.env = env
        self.cwd = cwd
        self.startup_timeout = startup_timeout
        self.proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 0
        self._reader: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr: deque[str] = deque(maxlen=60)
        self._write_lock = asyncio.Lock()
        self.list_changed = False

    async def start(self) -> None:
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
                cwd=self.cwd,
                start_new_session=True,
                limit=16 * 1024 * 1024,
            )
        except FileNotFoundError as exc:
            raise MCPError(
                f"cannot start MCP server: command not found: {self.command[0]}",
                error_class="unavailable",
                hint="Install the server or fix the command in the MCP configuration.",
            ) from exc
        self._reader = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                return
            self._stderr.append(line.decode("utf-8", errors="replace").rstrip())

    async def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self._stderr.append(f"[non-JSON stdout] {line[:200]!r}")
                    continue
                await self._dispatch(msg)
        finally:
            code = self.proc.returncode if self.proc else None
            err = MCPError(
                f"MCP server exited (code {code}); stderr: {' | '.join(list(self._stderr)[-5:])[:600]}",
                error_class="unavailable",
            )
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()

    async def _dispatch(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        if "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg:
            fut = self._pending.pop(msg["id"], None) if isinstance(msg["id"], int) else None
            if fut is not None and not fut.done():
                if "error" in msg:
                    fut.set_exception(_rpc_error(msg["error"]))
                else:
                    fut.set_result(msg.get("result") or {})
            return
        if "method" in msg and "id" in msg:
            if msg["method"] == "ping":
                await self._write({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
            else:
                await self._write(
                    {
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "error": {"code": -32601, "message": "Core Main does not handle server requests"},
                    }
                )
            return
        if msg.get("method") == "notifications/tools/list_changed":
            self.list_changed = True

    async def _write(self, msg: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None or self.proc.returncode is not None:
            raise MCPError("MCP server is not running", error_class="unavailable")
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        async with self._write_lock:
            self.proc.stdin.write(data)
            await self.proc.stdin.drain()

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        self._next_id += 1
        req_id = self._next_id
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError as exc:
            with contextlib.suppress(MCPError, OSError):
                await self.notify("notifications/cancelled", {"requestId": req_id, "reason": "timeout"})
            raise MCPError(
                f"MCP request {method} timed out after {timeout:.0f}s", error_class="timeout"
            ) from exc
        finally:
            self._pending.pop(req_id, None)

    async def notify(
        self, method: str, params: dict[str, Any] | None = None, *, headers: dict[str, str] | None = None
    ) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._write(msg)

    async def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
        for task in (self._reader, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    def diagnostics(self) -> str:
        return "\n".join(self._stderr)


class HttpTransport(Transport):
    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        *,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.url = url
        self.headers = headers
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            transport=transport,
            headers={"user-agent": f"core-main/{__version__}"},
        )
        self._next_id = 0

    def _headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        h = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        if self.protocol_version:
            h["mcp-protocol-version"] = self.protocol_version
        h.update(extra or {})
        return h

    async def request(
        self, method: str, params: dict[str, Any], *, timeout: float, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        self._next_id += 1
        req_id = self._next_id
        body = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        try:
            async with self._client.stream(
                "POST", self.url, json=body, headers=self._headers(headers), timeout=timeout
            ) as resp:
                if resp.status_code in (401, 403):
                    www = resp.headers.get("www-authenticate", "")
                    await resp.aread()
                    raise MCPError(
                        f"MCP server requires authorization (HTTP {resp.status_code})",
                        error_class="auth_required",
                        hint='Configure a token via secret_headers (e.g. Authorization = "store:NAME"); '
                        + (f"server metadata: {www[:200]}" if www else ""),
                    )
                sid = resp.headers.get("mcp-session-id")
                if sid:
                    self.session_id = sid
                ctype = resp.headers.get("content-type", "")
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="replace")
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and "error" in payload:
                            raise _rpc_error(payload["error"])
                    except json.JSONDecodeError:
                        pass
                    klass = (
                        "not_found"
                        if resp.status_code == 404
                        else ("unavailable" if resp.status_code >= 500 else "http_error")
                    )
                    raise MCPError(
                        f"MCP HTTP {resp.status_code}: {text[:300]}",
                        error_class=klass,
                        rpc_code=resp.status_code,
                    )
                if "text/event-stream" in ctype:
                    async for ev in iter_sse(resp.aiter_lines()):
                        if not ev.data:
                            continue
                        try:
                            msg = json.loads(ev.data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(msg, dict) and msg.get("id") == req_id:
                            if "error" in msg:
                                raise _rpc_error(msg["error"])
                            return msg.get("result") or {}
                    raise MCPError(
                        f"MCP stream ended without a response to {method}", error_class="invalid_response"
                    )
                text = (await resp.aread()).decode("utf-8", errors="replace")
        except httpx.TimeoutException as exc:
            raise MCPError(f"MCP request {method} timed out", error_class="timeout") from exc
        except httpx.HTTPError as exc:
            raise MCPError(
                f"cannot reach MCP server: {type(exc).__name__}: {exc}"[:300], error_class="unavailable"
            ) from exc
        try:
            msg = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MCPError(
                f"MCP server returned non-JSON for {method}", error_class="invalid_response"
            ) from exc
        if isinstance(msg, dict) and "error" in msg:
            raise _rpc_error(msg["error"])
        return (msg.get("result") if isinstance(msg, dict) else None) or {}

    async def notify(
        self, method: str, params: dict[str, Any] | None = None, *, headers: dict[str, str] | None = None
    ) -> None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        with contextlib.suppress(httpx.HTTPError):
            await self._client.post(self.url, json=body, headers=self._headers(headers))

    async def close(self) -> None:
        if self.session_id:
            with contextlib.suppress(httpx.HTTPError):
                await self._client.delete(self.url, headers=self._headers(None))
        await self._client.aclose()


@dataclass
class ServerInfo:
    era: str
    protocol_version: str
    server_info: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    instructions: str | None = None


class MCPClient:
    def __init__(self, name: str, transport: Transport, *, timeout: float, startup_timeout: float):
        self.name = name
        self.transport = transport
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.info: ServerInfo | None = None

    @property
    def modern(self) -> bool:
        return self.info is not None and self.info.era == "modern"

    def _params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(params or {})
        if self.modern:
            out["_meta"] = {**modern_meta(), **out.get("_meta", {})}
        return out

    def _headers(self, method: str, name: str | None = None) -> dict[str, str]:
        if not self.modern:
            return {}
        h = {"mcp-protocol-version": MODERN_VERSION, "mcp-method": method}
        if name:
            h["mcp-name"] = name
        return h

    async def connect(self) -> ServerInfo:
        await self.transport.start()
        probe_timeout = min(self.startup_timeout, 15.0)
        try:
            result = await self.transport.request(
                "server/discover",
                {"_meta": modern_meta()},
                timeout=probe_timeout,
                headers={"mcp-protocol-version": MODERN_VERSION, "mcp-method": "server/discover"},
            )
            versions = result.get("supportedVersions") or []
            if MODERN_VERSION in versions or not versions:
                meta = result.get("_meta") or {}
                self.info = ServerInfo(
                    "modern",
                    MODERN_VERSION,
                    meta.get(f"{META}/serverInfo") or result.get("serverInfo") or {},
                    result.get("capabilities") or {},
                    result.get("instructions"),
                )
                return self.info
        except MCPError as exc:
            if exc.error_class in {"unavailable", "auth_required"}:
                raise
            if exc.rpc_code in MODERN_ERRORS and isinstance(exc.data, dict):
                supported = exc.data.get("supported") or []
                if MODERN_VERSION in supported:
                    self.info = ServerInfo("modern", MODERN_VERSION)
                    return self.info
        return await self._legacy_initialize()

    async def _legacy_initialize(self) -> ServerInfo:
        result = await self.transport.request(
            "initialize",
            {"protocolVersion": LEGACY_VERSIONS[0], "capabilities": {}, "clientInfo": CLIENT_INFO},
            timeout=self.startup_timeout,
        )
        version = str(result.get("protocolVersion") or "")
        if version not in LEGACY_VERSIONS:
            raise MCPError(
                f"server negotiated unsupported protocol version '{version}'",
                error_class="unsupported",
                hint=f"Core Main supports {MODERN_VERSION} and {', '.join(LEGACY_VERSIONS)}.",
            )
        if isinstance(self.transport, HttpTransport):
            self.transport.protocol_version = version
        await self.transport.notify("notifications/initialized")
        self.info = ServerInfo(
            "legacy",
            version,
            result.get("serverInfo") or {},
            result.get("capabilities") or {},
            result.get("instructions"),
        )
        return self.info

    async def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = None
        for _ in range(50):
            params = self._params({"cursor": cursor} if cursor else {})
            result = await self.transport.request(
                "tools/list", params, timeout=self.timeout, headers=self._headers("tools/list")
            )
            tools.extend(t for t in result.get("tools", []) if isinstance(t, dict) and t.get("name"))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        result = await self.transport.request(
            "tools/call",
            self._params({"name": name, "arguments": arguments}),
            timeout=timeout or self.timeout,
            headers=self._headers("tools/call", name),
        )
        if result.get("resultType") == "input_required":
            raise MCPError(
                f"tool {name} requested interactive input (elicitation/sampling), which Core Main does not support",
                error_class="unsupported",
            )
        return result

    async def close(self) -> None:
        await self.transport.close()
