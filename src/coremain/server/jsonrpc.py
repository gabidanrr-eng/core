"""JSON-RPC 2.0 over stdio: the runtime API for editors and other clients (`core serve --stdio`).

Framing is detected from the first bytes: LSP-style ``Content-Length`` headers, or one JSON
message per line (NDJSON). Replies use the same framing. The server is a thin client of
``CoreRuntime``; it holds no domain state. Durable events are pushed as ``event``
notifications to clients that called ``events.subscribe``; streaming model text arrives as
``event`` notifications with ``ephemeral: true``.

Approvals never block the server: a task that needs one suspends durably
(``awaiting_approval``) and resumes after ``approval.decide``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from coremain.domain.states import ATTENTION, TERMINAL
from coremain.errors import CoreError, ExitCode, NotFoundError, UsageError
from coremain.version import __version__

if TYPE_CHECKING:
    from coremain.cli.common import CLIContext
    from coremain.events import Event
    from coremain.runtime.app import CoreRuntime

PROTOCOL_VERSION = 1
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = (
    -32700,
    -32600,
    -32601,
    -32602,
    -32603,
)
# Application errors: -32000 - exit code, so clients can map them like the CLI does.
APP_ERROR_BASE = -32000


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def _need(params: dict[str, Any], key: str) -> Any:
    if key not in params:
        raise RpcError(INVALID_PARAMS, f"missing parameter '{key}'")
    return params[key]


class Transport:
    """Reads and writes framed JSON messages; framing is fixed by the first message."""

    def __init__(self, reader: asyncio.StreamReader, write: Callable[[bytes], None]):
        self.reader = reader
        self._write = write
        self.framing: str | None = None
        self._lock = asyncio.Lock()

    async def read(self) -> bytes | None:
        while True:
            line = await self.reader.readline()
            if not line:
                return None
            if self.framing is None:
                self.framing = "lsp" if line.lower().startswith(b"content-length:") else "ndjson"
            if self.framing == "ndjson":
                if line.strip():
                    return line
                continue
            length = None
            while line.strip():
                name, _, value = line.decode("ascii", errors="replace").partition(":")
                if name.strip().lower() == "content-length":
                    length = int(value.strip())
                line = await self.reader.readline()
                if not line:
                    return None
            if length is None:
                continue
            return await self.reader.readexactly(length)

    async def send(self, message: dict[str, Any]) -> None:
        body = json.dumps(message, ensure_ascii=False, default=str).encode("utf-8")
        frame = (
            (f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
            if self.framing == "lsp"
            else body + b"\n"
        )
        async with self._lock:
            self._write(frame)


class Server:
    def __init__(self, rt: CoreRuntime, transport: Transport):
        self.rt = rt
        self.transport = transport
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stop = asyncio.Event()
        self._methods: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {
            "initialize": self.initialize,
            "shutdown": self.shutdown,
            "status": self.status,
            "session.list": self.session_list,
            "session.create": self.session_create,
            "session.messages": self.session_messages,
            "task.submit": self.task_submit,
            "task.get": self.task_get,
            "task.list": self.task_list,
            "task.cancel": self.task_cancel,
            "task.answer": self.task_answer,
            "task.resume": self.task_resume,
            "task.diff": self.task_diff,
            "task.evidence": self.task_evidence,
            "task.events": self.task_events,
            "task.wait": self.task_wait,
            "approval.list": self.approval_list,
            "approval.decide": self.approval_decide,
            "memory.search": self.memory_search,
            "index.search": self.index_search,
            "events.subscribe": self.events_subscribe,
            "events.unsubscribe": self.events_unsubscribe,
        }

    # ------------------------------------------------------------------ loop
    async def serve(self) -> None:
        pump = asyncio.create_task(self._pump_events())
        stopper = asyncio.create_task(self._stop.wait())
        try:
            while not self._stop.is_set():
                reader = asyncio.create_task(self.transport.read())
                await asyncio.wait({reader, stopper}, return_when=asyncio.FIRST_COMPLETED)
                if not reader.done():
                    reader.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await reader
                    break
                raw = reader.result()
                if raw is None:
                    break
                task = asyncio.create_task(self._handle_raw(raw))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        finally:
            for helper in (pump, stopper):
                helper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await helper
            for task in list(self._tasks):
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(task, timeout=5)

    async def _handle_raw(self, raw: bytes) -> None:
        try:
            message = json.loads(raw)
        except ValueError as exc:
            await self.transport.send(
                {"jsonrpc": "2.0", "id": None, "error": {"code": PARSE_ERROR, "message": str(exc)}}
            )
            return
        batch = message if isinstance(message, list) else [message]
        replies = [r for r in [await self._handle(m) for m in batch] if r is not None]
        if isinstance(message, list) and replies:
            for reply in replies:
                await self.transport.send(reply)
        elif replies:
            await self.transport.send(replies[0])

    async def _handle(self, msg: Any) -> dict[str, Any] | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return {
                "jsonrpc": "2.0",
                "id": msg.get("id") if isinstance(msg, dict) else None,
                "error": {"code": INVALID_REQUEST, "message": "invalid JSON-RPC 2.0 request"},
            }
        is_notification = "id" not in msg
        try:
            handler = self._methods.get(msg["method"])
            if handler is None:
                raise RpcError(METHOD_NOT_FOUND, f"unknown method {msg['method']}")
            params = msg.get("params") or {}
            if not isinstance(params, dict):
                raise RpcError(INVALID_PARAMS, "params must be an object")
            result = await handler(params)
            return None if is_notification else {"jsonrpc": "2.0", "id": msg["id"], "result": result}
        except RpcError as exc:
            error = {
                "code": exc.code,
                "message": exc.message,
                **({"data": exc.data} if exc.data is not None else {}),
            }
        except CoreError as exc:
            error = {
                "code": APP_ERROR_BASE - int(exc.exit_code),
                "message": exc.message,
                "data": {"error": exc.code, "hint": getattr(exc, "hint", None)},
            }
        except Exception as exc:  # noqa: BLE001 - never kill the server on a handler bug
            error = {"code": INTERNAL_ERROR, "message": f"{type(exc).__name__}: {exc}"}
        return None if is_notification else {"jsonrpc": "2.0", "id": msg.get("id"), "error": error}

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self.transport.send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _pump_events(self) -> None:
        """Durable events (including other processes') by tailing the log; ephemeral ones from the bus."""
        seq = self.rt.events.latest_seq()
        sub = self.rt.bus.subscribe(lambda e: not e.durable)
        try:
            while True:
                if self.subscriptions:
                    for ev in self.rt.events.since(seq, limit=500):
                        seq = ev.seq or seq
                        await self._deliver(ev, ephemeral=False)
                    while (ev := sub.get_nowait()) is not None:
                        await self._deliver(ev, ephemeral=True)
                else:
                    seq = self.rt.events.latest_seq()
                    while sub.get_nowait() is not None:
                        pass
                await asyncio.sleep(0.2)
        finally:
            sub.close()

    async def _deliver(self, ev: Event, *, ephemeral: bool) -> None:
        for sub_id, spec in list(self.subscriptions.items()):
            if spec.get("task_id") and ev.task_id != spec["task_id"]:
                continue
            if spec.get("kinds") and not any(ev.kind.startswith(k) for k in spec["kinds"]):
                continue
            if ephemeral and not spec.get("ephemeral", True):
                continue
            await self.notify("event", {"subscription": sub_id, "ephemeral": ephemeral, **ev.to_dict()})

    # --------------------------------------------------------------- methods
    async def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        project = self.rt.project
        return {
            "server": "core-main",
            "version": __version__,
            "protocol": PROTOCOL_VERSION,
            "framing": self.transport.framing,
            "project": {"id": project.id, "root": project.root_path, "name": project.name}
            if project
            else None,
            "methods": sorted(self._methods),
        }

    async def shutdown(self, params: dict[str, Any]) -> dict[str, Any]:
        self._stop.set()
        return {"ok": True}

    async def status(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.rt.status()

    async def session_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        project = self.rt.require_project()
        return [s.__dict__ for s in self.rt.sessions.list(project.id, limit=int(params.get("limit", 50)))]

    async def session_create(self, params: dict[str, Any]) -> dict[str, Any]:
        project = self.rt.require_project()
        return self.rt.sessions.create(project.id, params.get("title") or "New session").__dict__

    async def session_messages(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        session = self.rt.sessions.resolve(
            _need(params, "session_id"), project_id=self.rt.require_project().id
        )
        return [
            m.__dict__ for m in self.rt.sessions.messages(session.id, limit=int(params.get("limit", 200)))
        ]

    async def task_submit(self, params: dict[str, Any]) -> dict[str, Any]:
        task = await self.rt.submit(
            str(_need(params, "text")),
            session_id=params.get("session_id"),
            mode=params.get("mode"),
            model=params.get("model"),
            auto_apply=params.get("auto_apply"),
            contract=params.get("contract"),
        )
        if params.get("start", True):
            self.rt.start_task(task.id)
        return task.to_dict()

    def _task(self, params: dict[str, Any]) -> Any:
        return self.rt.tasks.resolve(
            str(_need(params, "task_id")), project_id=self.rt.project.id if self.rt.project else None
        )

    async def task_get(self, params: dict[str, Any]) -> dict[str, Any]:
        task = self._task(params)
        data = task.to_dict()
        data["attempts"] = [a.to_dict() for a in self.rt.tasks.attempts(task.id)]
        data["pending_approvals"] = [a.to_dict() for a in self.rt.approvals.pending(task_id=task.id)]
        return data

    async def task_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        project = self.rt.require_project()
        kwargs: dict[str, Any] = {"project_id": project.id, "limit": int(params.get("limit", 50))}
        if params.get("session_id"):
            kwargs["session_id"] = params["session_id"]
        tasks = self.rt.tasks.list(**kwargs)
        if params.get("status"):
            tasks = [t for t in tasks if t.status.value == params["status"]]
        return [t.to_dict() for t in tasks]

    async def task_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.rt.cancel_task(
            self._task(params).id, reason=params.get("reason") or "cancelled by client"
        ).to_dict()

    async def task_answer(self, params: dict[str, Any]) -> dict[str, Any]:
        task = self.rt.answer(self._task(params).id, str(_need(params, "text")))
        if params.get("start", True) and task.status.value == "queued":
            self.rt.start_task(task.id)
        return task.to_dict()

    async def task_resume(self, params: dict[str, Any]) -> dict[str, Any]:
        task = self._task(params)
        if task.status.value != "queued":
            task = self.rt.tasks.resume(task.id, note="resumed by client")
        self.rt.start_task(task.id)
        return task.to_dict()

    async def task_diff(self, params: dict[str, Any]) -> dict[str, Any]:
        task = self._task(params)
        if not task.workspace_id:
            raise NotFoundError(f"task {task.id} has no workspace")
        ws = self.rt.workspaces.get(task.workspace_id)
        if not ws.path.exists():
            raise NotFoundError(f"workspace {ws.id} no longer exists on disk ({ws.status})")
        diff = await self.rt.workspaces.diff(ws)
        return {
            "files": [f.__dict__ for f in diff.files],
            "stats": diff.stats,
            "diff_hash": diff.diff_hash,
            "patch": diff.patch,
        }

    async def task_evidence(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.rt.evidence.for_task(self._task(params).id)]

    async def task_events(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        task = self._task(params)
        since = int(params.get("since", 0))
        return [e.to_dict() for e in self.rt.events.for_task(task.id) if (e.seq or 0) > since][
            : int(params.get("limit", 1000))
        ]

    async def task_wait(self, params: dict[str, Any]) -> dict[str, Any]:
        """Wait until the task is terminal or needs attention (bounded by ``timeout_s``)."""
        task = self._task(params)
        timeout = float(params.get("timeout_s", 300))
        done = {s.value for s in TERMINAL | ATTENTION}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            task = self.rt.tasks.get(task.id)
            if task.status.value in done or loop.time() >= deadline:
                return {**task.to_dict(), "timed_out": task.status.value not in done}
            await asyncio.sleep(0.2)

    async def approval_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        project_id = self.rt.project.id if self.rt.project else None
        return [
            a.to_dict()
            for a in self.rt.approvals.pending(project_id=project_id, task_id=params.get("task_id"))
        ]

    async def approval_decide(self, params: dict[str, Any]) -> dict[str, Any]:
        approved = _need(params, "approved")
        if not isinstance(approved, bool):
            raise RpcError(INVALID_PARAMS, "'approved' must be a boolean")
        approval = self.rt.approvals.resolve(str(_need(params, "approval_id")))
        decided = self.rt.decide_approval(
            approval.id, approved, scope=params.get("scope", "once"), reason=params.get("reason")
        )
        if decided.task_id:
            task = self.rt.tasks.get(decided.task_id)
            if task.status.value == "queued":
                self.rt.start_task(task.id)
        return decided.to_dict()

    async def memory_search(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        project = self.rt.require_project()
        hits = self.rt.memory.search(
            project.id, str(_need(params, "query")), limit=int(params.get("limit", 20))
        )
        return [h.to_dict() if hasattr(h, "to_dict") else dict(h.__dict__) for h in hits]

    async def index_search(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        project = self.rt.require_project()
        hits = self.rt.index.search(
            project.id, str(_need(params, "query")), limit=int(params.get("limit", 20))
        )
        return [dict(h.__dict__) for h in hits]

    async def events_subscribe(self, params: dict[str, Any]) -> dict[str, Any]:
        from coremain.util.ids import new_id

        sub_id = new_id("sub")
        kinds = params.get("kinds")
        if kinds is not None and not (isinstance(kinds, list) and all(isinstance(k, str) for k in kinds)):
            raise RpcError(INVALID_PARAMS, "'kinds' must be a list of event kind prefixes")
        self.subscriptions[sub_id] = {
            "task_id": params.get("task_id"),
            "kinds": kinds,
            "ephemeral": bool(params.get("ephemeral", True)),
        }
        return {"subscription": sub_id}

    async def events_unsubscribe(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"removed": self.subscriptions.pop(str(_need(params, "subscription")), None) is not None}


async def _stdio() -> tuple[asyncio.StreamReader, Callable[[bytes], None]]:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=64 * 1024 * 1024)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    out = sys.stdout.buffer

    def write(data: bytes) -> None:
        out.write(data)
        out.flush()

    return reader, write


async def serve_stdio(ctx: CLIContext) -> None:
    if ctx.output.json_mode is False and sys.stdout.isatty():
        raise UsageError(
            "`core serve --stdio` speaks JSON-RPC on stdin/stdout; run it from a client, not a terminal",
            hint="see docs/json-rpc.md",
        )
    rt = ctx.open_runtime(mode="serve", require_project=False, detect_project=True)
    try:
        await rt.start()
        reader, write = await _stdio()
        await Server(rt, Transport(reader, write)).serve()
    finally:
        await rt.close()


__all__ = ["APP_ERROR_BASE", "ExitCode", "RpcError", "Server", "Transport", "serve_stdio"]
