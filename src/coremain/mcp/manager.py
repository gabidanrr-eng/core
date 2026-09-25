"""MCP server lifecycle, cached discovery state and policy-governed tool adapters."""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict

from coremain.config.schema import MCPServerConfig
from coremain.errors import NotFoundError
from coremain.mcp.client import HttpTransport, MCPClient, MCPError, StdioTransport, Transport
from coremain.security.credentials import resolve_credential
from coremain.security.env import build_subprocess_env
from coremain.security.policy import Capability, PolicyRequest
from coremain.tools.base import SideEffect, Tool, ToolContext, ToolInput, ToolResult
from coremain.util.jsonutil import dumps, loads, stable_hash

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime

STATE_TTL_S = 600.0
READ_ONLY_ROLES = {"planner", "researcher", "reviewer"}


def tool_name(server: str, tool: str) -> str:
    raw = f"mcp__{server}__{tool}"
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    return clean[:64]


def _matches(name: str, patterns: list[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(name, p) for p in patterns)


class MCPManager:
    def __init__(self, rt: CoreRuntime):
        self.rt = rt
        self._clients: dict[str, MCPClient] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _cfg(self, name: str) -> MCPServerConfig:
        cfg = self.rt.config.mcp.get(name)
        if cfg is None:
            raise NotFoundError(f"MCP server '{name}' is not configured", hint="Add it with `core mcp add`.")
        return cfg

    def _transport(self, name: str, cfg: MCPServerConfig) -> Transport:
        store = self.rt.credentials
        if cfg.transport == "stdio":
            env = build_subprocess_env(self.rt.env, passthrough=self.rt.config.permissions.env_passthrough)
            env.update(cfg.env)
            for key, ref in cfg.secret_env.items():
                value = resolve_credential(ref, store, env=self.rt.env, redactor=self.rt.redactor).value
                if value:
                    env[key] = value
            return StdioTransport(
                list(cfg.command),
                env,
                cwd=str(self.rt.project_root) if self.rt.project_root else None,
                startup_timeout=cfg.startup_timeout_s,
            )
        headers = dict(cfg.headers)
        for header, ref in cfg.secret_headers.items():
            value = resolve_credential(ref, store, env=self.rt.env, redactor=self.rt.redactor).value
            if value:
                headers[header] = (
                    value if header.lower() != "authorization" or " " in value else f"Bearer {value}"
                )
        return HttpTransport(str(cfg.url), headers, timeout=cfg.timeout_s)

    async def client(self, name: str) -> MCPClient:
        cfg = self._cfg(name)
        if not cfg.enabled:
            raise MCPError(f"MCP server '{name}' is disabled", error_class="unavailable")
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            existing = self._clients.get(name)
            if existing is not None:
                return existing
            if cfg.transport == "http" and self.rt.config.permissions.network.offline:
                raise MCPError("offline mode: remote MCP servers are disabled", error_class="unavailable")
            client = MCPClient(
                name, self._transport(name, cfg), timeout=cfg.timeout_s, startup_timeout=cfg.startup_timeout_s
            )
            try:
                await client.connect()
            except BaseException:
                with contextlib.suppress(Exception):
                    await client.close()
                raise
            self._clients[name] = client
            return client

    def _save_state(self, name: str, **fields: Any) -> None:
        cfg = self.rt.config.mcp.get(name)
        self.rt.db.execute(
            "INSERT INTO mcp_state(server, config_hash, status, error_class, error_message, protocol_version, server_info_json, tools_json, checked_at) "
            "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(server) DO UPDATE SET config_hash = excluded.config_hash, status = excluded.status, "
            "error_class = excluded.error_class, error_message = excluded.error_message, protocol_version = excluded.protocol_version, "
            "server_info_json = excluded.server_info_json, tools_json = excluded.tools_json, checked_at = excluded.checked_at",
            (
                name,
                stable_hash(cfg.model_dump(mode="json")) if cfg else None,
                fields.get("status"),
                fields.get("error_class"),
                fields.get("error_message"),
                fields.get("protocol_version"),
                dumps(fields.get("server_info") or {}),
                dumps(fields.get("tools") or []),
                self.rt.clock.now(),
            ),
        )

    async def check(self, name: str) -> dict[str, Any]:
        await self._drop(name)
        try:
            client = await self.client(name)
            tools = await client.list_tools()
            info = client.info
            assert info is not None
            self._save_state(
                name,
                status="ready",
                protocol_version=f"{info.era}:{info.protocol_version}",
                server_info=info.server_info,
                tools=tools,
            )
            self.rt.events.emit(
                "mcp.ready",
                data={
                    "server": name,
                    "protocol": info.protocol_version,
                    "era": info.era,
                    "tools": len(tools),
                },
            )
            return {
                "name": name,
                "status": "ready",
                "protocol_version": info.protocol_version,
                "era": info.era,
                "tools": len(tools),
                "server_info": info.server_info,
            }
        except MCPError as exc:
            diag = ""
            client = self._clients.get(name)
            if client is not None:
                diag = client.transport.diagnostics()
            self._save_state(
                name, status="error", error_class=exc.error_class, error_message=exc.message[:500]
            )
            self.rt.events.emit(
                "mcp.unavailable",
                level="warning",
                data={"server": name, "error_class": exc.error_class, "message": exc.message[:300]},
            )
            await self._drop(name)
            return {
                "name": name,
                "status": "error",
                "error_class": exc.error_class,
                "error": self.rt.redactor.redact(exc.message),
                "hint": exc.hint,
                "stderr": self.rt.redactor.redact(diag[-800:]),
            }

    async def ensure_ready(self) -> None:
        """Refresh stale discovery state for enabled servers (called at task start)."""
        now = self.rt.clock.now()
        for name, cfg in self.rt.config.mcp.items():
            if not cfg.enabled:
                continue
            row = self.rt.db.one(
                "SELECT status, checked_at, config_hash FROM mcp_state WHERE server = ?", (name,)
            )
            fresh = (
                row is not None
                and now - float(row["checked_at"]) < STATE_TTL_S
                and row["config_hash"] == stable_hash(cfg.model_dump(mode="json"))
            )
            if not fresh:
                await self.check(name)

    async def list_tools(self, name: str) -> list[dict[str, Any]]:
        client = await self.client(name)
        return await client.list_tools()

    async def call_tool(self, name: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        cfg = self._cfg(name)
        client = await self.client(name)
        try:
            result = await client.call_tool(tool, arguments, timeout=cfg.timeout_s)
        except MCPError as exc:
            if exc.error_class in {"unavailable"}:
                await self._drop(name)
            raise
        return self.normalize_result(result)

    def normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        texts: list[str] = []
        images: list[tuple[str, str]] = []
        for block in result.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                texts.append(str(block.get("text", "")))
            elif btype == "image" and block.get("data"):
                images.append((block.get("mimeType", "image/png"), block["data"]))
            elif btype == "resource":
                res = block.get("resource") or {}
                texts.append(f"[resource {res.get('uri', '')}]\n{res.get('text', '(binary)')}")
            elif btype == "resource_link":
                texts.append(f"[resource link {block.get('uri', '')} {block.get('name', '')}]")
            elif btype == "audio":
                texts.append("[audio content omitted]")
        if result.get("structuredContent") is not None:
            texts.append("structured: " + dumps(result["structuredContent"])[:20000])
        return {"text": "\n".join(texts), "images": images, "is_error": bool(result.get("isError"))}

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for name, cfg in self.rt.config.mcp.items():
            st = self.rt.db.one("SELECT * FROM mcp_state WHERE server = ?", (name,))
            rows.append(
                {
                    "name": name,
                    "transport": cfg.transport,
                    "trust": cfg.trust,
                    "enabled": cfg.enabled,
                    "status": st["status"] if st else "unchecked",
                    "tools": len(loads(st["tools_json"], [])) if st else 0,
                    "checked_at": st["checked_at"] if st else None,
                    "error": st["error_message"] if st else None,
                    "protocol": st["protocol_version"] if st else None,
                }
            )
        return rows

    def tools_for_role(self, role: str) -> list[Tool]:
        out: list[Tool] = []
        for name, cfg in self.rt.config.mcp.items():
            if not cfg.enabled:
                continue
            st = self.rt.db.one("SELECT status, tools_json FROM mcp_state WHERE server = ?", (name,))
            if st is None or st["status"] != "ready":
                continue
            for spec in loads(st["tools_json"], []):
                tname = spec.get("name", "")
                if not _matches(tname, cfg.tool_allow) or _matches(tname, cfg.tool_deny):
                    continue
                annotations = spec.get("annotations") or {}
                if role in READ_ONLY_ROLES and not annotations.get("readOnlyHint"):
                    continue
                out.append(MCPToolAdapter(self, name, cfg, spec))
        return out

    async def _drop(self, name: str) -> None:
        client = self._clients.pop(name, None)
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()

    async def aclose(self) -> None:
        for name in list(self._clients):
            await self._drop(name)


class _MCPArgs(ToolInput):
    model_config = ConfigDict(extra="allow")


_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class MCPToolAdapter(Tool):
    capability = Capability.MCP
    timeout_s = 600.0
    Input = _MCPArgs

    def __init__(self, manager: MCPManager, server: str, cfg: MCPServerConfig, spec: dict[str, Any]):
        self.manager = manager
        self.server = server
        self.cfg = cfg
        self.spec_data = spec
        self.remote_name = spec["name"]
        self.name = tool_name(server, self.remote_name)  # type: ignore[misc]
        desc = spec.get("description") or spec.get("title") or self.remote_name
        self.description = f"[MCP server '{server}', {cfg.trust}] {desc}"[:1024]  # type: ignore[misc]
        self.annotations = spec.get("annotations") or {}
        self.read_only = bool(self.annotations.get("readOnlyHint"))  # type: ignore[misc]
        if self.read_only:
            self.side_effect = SideEffect.NONE  # type: ignore[misc]
        elif self.annotations.get("destructiveHint", True):
            self.side_effect = SideEffect.DESTRUCTIVE  # type: ignore[misc]
        elif self.annotations.get("openWorldHint", True):
            self.side_effect = SideEffect.EXTERNAL  # type: ignore[misc]
        else:
            self.side_effect = SideEffect.LOCAL  # type: ignore[misc]
        self.timeout_s = cfg.timeout_s + 5  # type: ignore[misc]

    def schema(self) -> dict[str, Any]:  # type: ignore[override]
        schema = dict(self.spec_data.get("inputSchema") or {})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema

    def spec(self) -> dict[str, Any]:  # type: ignore[override]
        return {"name": self.name, "description": self.description, "parameters": self.schema()}

    def target(self, args: Any) -> str:
        return f"{self.server}:{self.remote_name}"

    def policy_request(self, ctx: ToolContext, args: Any) -> PolicyRequest | None:
        return PolicyRequest(
            capability=Capability.MCP,
            target=self.target(args),
            workspace_root=ctx.workspace.path,
            workspace_isolated=ctx.workspace.isolated,
            mcp_annotations=self.annotations,
            mcp_trusted=self.cfg.trust == "trusted",
            project_id=ctx.project_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            tool=self.name,
        )

    def _validate(self, arguments: dict[str, Any]) -> str | None:
        schema = self.schema()
        for key in schema.get("required", []) or []:
            if key not in arguments:
                return f"missing required argument '{key}'"
        for key, value in arguments.items():
            prop = (schema.get("properties") or {}).get(key)
            if not isinstance(prop, dict):
                continue
            expected = prop.get("type")
            types = expected if isinstance(expected, list) else [expected]
            py: tuple[type, ...] = ()
            for name in types:
                mapped = _JSON_TYPES.get(name) if isinstance(name, str) else None
                if mapped is not None:
                    py += mapped if isinstance(mapped, tuple) else (mapped,)
            if py and value is not None and not isinstance(value, py):
                return f"argument '{key}' should be {expected}"
        return None

    async def run(self, ctx: ToolContext, args: Any) -> ToolResult:
        arguments = args.model_dump(exclude_unset=False) if hasattr(args, "model_dump") else dict(args)
        problem = self._validate(arguments)
        if problem:
            return ToolResult.error(f"invalid arguments for {self.name}: {problem}", "invalid_arguments")
        try:
            result = await self.manager.call_tool(self.server, self.remote_name, arguments)
        except MCPError as exc:
            return ToolResult.error(f"MCP {exc.error_class}: {exc.message}", f"mcp_{exc.error_class}")
        text = f"[untrusted output from MCP server '{self.server}']\n" + (
            result["text"] or "(no text content)"
        )
        return ToolResult(
            not result["is_error"],
            text,
            status="ok" if not result["is_error"] else "error",
            error_class="mcp_tool_error" if result["is_error"] else None,
            images=result["images"],
        )
