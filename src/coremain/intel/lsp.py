"""Language Server Protocol integration: compiler-accurate navigation and diagnostics for agents.

``LSPManager`` is the ``lsp`` runtime extension. It discovers servers from ``intel.lsp_servers``
plus built-in defaults that are used only when their executable is on ``PATH`` (configured servers
are authoritative for the languages they list), runs one server process per (server, root) and
offers five read-only tools to every agent role.

* Transport: JSON-RPC 2.0 over stdio with ``Content-Length`` framing. Every request has a timeout;
  timed-out or cancelled requests are followed by ``$/cancelRequest``. Requests from the server get
  conservative answers (no configuration, workspace edits refused).
* Processes run in their own process group with the minimal subprocess environment, are recorded
  in the ``processes`` table so recovery can reap them after a crash, and are stopped with
  ``shutdown``/``exit`` followed by SIGTERM/SIGKILL of the whole group.
* A crashed server fails its pending requests with ``lsp_crashed``; the next use restarts it with
  exponential backoff until the restart budget is exhausted.
* Documents are synced from disk before every query (full-text ``didOpen``/``didChange`` with
  increasing versions), so answers describe the workspace as it is now.
* Failures carry stable error classes (``lsp_unavailable``, ``lsp_timeout``, ``lsp_crashed``, ...).
  Results are never synthesized when a server cannot answer.

Tool positions are 1-based lines and 1-based columns counted in characters (code points), the same
convention as ``read_file`` line numbers and the ``path:line:col`` results; conversion to the
negotiated LSP position encoding (UTF-16 by default) happens here.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import sqlite3
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from pydantic import Field, model_validator

from coremain.errors import CoreError, ExitCode, ToolError
from coremain.exec.process import _start_ticks, kill_group
from coremain.intel.languages import CODE_LANGUAGES, detect_language
from coremain.security.env import build_subprocess_env
from coremain.security.paths import sensitive_reason
from coremain.security.policy import Capability
from coremain.tools.base import Tool, ToolContext, ToolInput, ToolResult
from coremain.tools.fs import IGNORED_DIRS, guard
from coremain.util.ids import new_id
from coremain.util.jsonutil import sha256_hex
from coremain.util.text import one_line, truncate_middle
from coremain.version import __version__

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime

log = logging.getLogger(__name__)

LSP_UNAVAILABLE = "lsp_unavailable"
LSP_TIMEOUT = "lsp_timeout"
LSP_CRASHED = "lsp_crashed"
LSP_PROTOCOL = "lsp_protocol_error"
LSP_UNSUPPORTED = "lsp_unsupported"
LSP_REQUEST_FAILED = "lsp_request_failed"
LSP_CANCELLED = "lsp_cancelled"
LSP_CONTENT_MODIFIED = "lsp_content_modified"
LSP_FILE_TOO_LARGE = "lsp_file_too_large"

FALLBACK_HINT = "fall back to find_symbol, find_references, file_outline or search_text"
ROLES = frozenset({"planner", "implementer", "corrector", "debugger", "researcher", "reviewer"})
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_SYNC_BYTES = 4 * 1024 * 1024
MAX_HEADER_LINES = 32
STDERR_LINES = 200
STDERR_LINE_CHARS = 500
MAX_BACKOFF_S = 30.0
STABLE_UPTIME_S = 300.0
LANGUAGE_SCAN_LIMIT = 5000
PREVIEW_CHARS = 160
MAX_HOVER_CHARS = 6000
MAX_SYMBOLS = 400
MAX_DIAGNOSTICS = 100
MAX_DEFINITIONS = 50
MAX_STORED_DIAGNOSTICS = 2000
MAX_DIAGNOSTIC_URIS = 2000

_RPC_ERROR_CLASSES = {
    -32601: LSP_UNSUPPORTED,  # MethodNotFound
    -32800: LSP_CANCELLED,  # RequestCancelled
    -32801: LSP_CONTENT_MODIFIED,  # ContentModified
    -32802: LSP_CANCELLED,  # ServerCancelled
}
_RETRYABLE_RPC_CODES = frozenset({-32801, -32802})
SYMBOL_KINDS = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    19: "object",
    20: "key",
    21: "null",
    22: "enum member",
    23: "struct",
    24: "event",
    25: "operator",
    26: "type parameter",
}
# A missing severity is shown as an error, as VS Code does.
SEVERITIES = {1: "error", 2: "warning", 3: "info", 4: "hint"}
_LANGUAGE_IDS = {"shell": "shellscript", "docker": "dockerfile", "make": "makefile"}
_SUFFIX_LANGUAGE_IDS = {".jsx": "javascriptreact", ".tsx": "typescriptreact", ".scss": "scss"}

CLIENT_CAPABILITIES: dict[str, Any] = {
    "general": {"positionEncodings": ["utf-16", "utf-32", "utf-8"]},
    "workspace": {
        "configuration": True,
        "workspaceFolders": True,
        "applyEdit": False,
        "didChangeConfiguration": {"dynamicRegistration": False},
        "didChangeWatchedFiles": {"dynamicRegistration": False},
    },
    "window": {"workDoneProgress": True, "showDocument": {"support": False}},
    "textDocument": {
        "synchronization": {
            "dynamicRegistration": False,
            "willSave": False,
            "willSaveWaitUntil": False,
            "didSave": False,
        },
        "definition": {"dynamicRegistration": False, "linkSupport": True},
        "references": {"dynamicRegistration": False},
        "hover": {"dynamicRegistration": False, "contentFormat": ["markdown", "plaintext"]},
        "documentSymbol": {
            "dynamicRegistration": False,
            "hierarchicalDocumentSymbolSupport": True,
            "symbolKind": {"valueSet": list(range(1, 27))},
        },
        "publishDiagnostics": {
            "relatedInformation": False,
            "versionSupport": True,
            "tagSupport": {"valueSet": [1, 2]},
        },
        "diagnostic": {"dynamicRegistration": True, "relatedDocumentSupport": False},
    },
}


class LSPError(ToolError):
    """A language-server failure with a stable, machine-readable ``error_class``."""

    code = "lsp_error"

    def __init__(
        self, message: str, *, error_class: str, rpc_code: int | None = None, hint: str | None = None
    ):
        super().__init__(
            message,
            error_class=error_class,
            hint=hint,
            details={"rpc_code": rpc_code} if rpc_code is not None else None,
        )
        self.rpc_code = rpc_code
        self.exit_code = (
            ExitCode.BLOCKED if error_class in (LSP_UNAVAILABLE, LSP_CRASHED) else ExitCode.FAILURE
        )

    def copy(self) -> LSPError:
        return LSPError(self.message, error_class=self.error_class, rpc_code=self.rpc_code, hint=self.hint)


def _rpc_error(server: str, method: str, err: Mapping[str, Any]) -> LSPError:
    code = err.get("code")
    rpc_code = code if isinstance(code, int) else None
    klass = (
        _RPC_ERROR_CLASSES.get(rpc_code, LSP_REQUEST_FAILED) if rpc_code is not None else LSP_REQUEST_FAILED
    )
    message = one_line(str(err.get("message") or "error"), 300)
    return LSPError(
        f"language server '{server}' failed {method}: {message} (code {code})",
        error_class=klass,
        rpc_code=rpc_code,
    )


# ---------------------------------------------------------------------------------- framing
def encode_message(msg: Mapping[str, Any]) -> bytes:
    body = json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return b"Content-Length: %d\r\n\r\n" % len(body) + body


async def read_message(
    reader: asyncio.StreamReader,
    *,
    max_bytes: int = MAX_MESSAGE_BYTES,
    noise: Callable[[str], None] | None = None,
) -> Any | None:
    """Read one framed message. Returns ``None`` on a clean end of stream between messages.

    Lines without a colon before a header block are stray output (reported through ``noise``);
    anything else malformed raises ``LSPError(lsp_protocol_error)``.
    """
    length: int | None = None
    headers = 0
    while True:
        try:
            line = await reader.readline()
        except ValueError as exc:
            raise LSPError("server wrote an oversized header line", error_class=LSP_PROTOCOL) from exc
        if not line:
            if headers == 0:
                return None
            raise LSPError("server closed its output in the middle of a message", error_class=LSP_PROTOCOL)
        stripped = line.strip()
        if not stripped:
            if length is not None:
                break
            if headers:
                raise LSPError("message header without Content-Length", error_class=LSP_PROTOCOL)
            continue
        name, sep, value = stripped.decode("ascii", errors="replace").partition(":")
        if not sep:
            if headers == 0:
                if noise is not None:
                    noise(stripped.decode("utf-8", errors="replace"))
                continue
            raise LSPError(f"malformed header line {one_line(name, 80)!r}", error_class=LSP_PROTOCOL)
        headers += 1
        if headers > MAX_HEADER_LINES:
            raise LSPError("too many header lines", error_class=LSP_PROTOCOL)
        if name.strip().lower() == "content-length":
            try:
                length = int(value.strip())
            except ValueError as exc:
                raise LSPError(
                    f"invalid Content-Length {one_line(value.strip(), 40)!r}", error_class=LSP_PROTOCOL
                ) from exc
            if length < 0 or length > max_bytes:
                raise LSPError(
                    f"message of {length} bytes exceeds the {max_bytes} byte limit", error_class=LSP_PROTOCOL
                )
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise LSPError(
            "server closed its output in the middle of a message", error_class=LSP_PROTOCOL
        ) from exc
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LSPError(f"server sent invalid JSON: {exc}", error_class=LSP_PROTOCOL) from exc


# -------------------------------------------------------------------- positions and URIs
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def split_lines(text: str) -> list[str]:
    """Split exactly on LSP line terminators (``str.splitlines`` also splits on \\f, \\x85, ...)."""
    return _LINE_BREAK.split(text)


def to_lsp_character(line_text: str, column: int, encoding: str) -> int:
    """Convert a code-point column into the negotiated position encoding."""
    prefix = line_text[:column]
    if encoding == "utf-32":
        return len(prefix)
    if encoding == "utf-8":
        return len(prefix.encode("utf-8", errors="surrogatepass"))
    return len(prefix.encode("utf-16-le", errors="surrogatepass")) // 2


def from_lsp_character(line_text: str, character: int, encoding: str) -> int:
    """Convert a position in the negotiated encoding back to a code-point column."""
    if character <= 0:
        return 0
    if encoding == "utf-32":
        return min(character, len(line_text))
    units = 0
    for index, ch in enumerate(line_text):
        if units >= character:
            return index
        code = ord(ch)
        if encoding == "utf-8":
            units += 1 if code < 0x80 else 2 if code < 0x800 else 3 if code < 0x10000 else 4
        else:
            units += 2 if code > 0xFFFF else 1
    return len(line_text)


def uri_to_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    return Path(unquote(parsed.path))


def normalize_uri(uri: str) -> str:
    """Canonical form for comparing URIs (servers differ in percent-encoding)."""
    path = uri_to_path(uri)
    return path.as_uri() if path is not None and path.is_absolute() else uri


def language_id(rel: str, language: str) -> str:
    return _SUFFIX_LANGUAGE_IDS.get(PurePosixPath(rel).suffix.lower()) or _LANGUAGE_IDS.get(
        language, language
    )


def lsp_root(workspace_root: Path, file_path: Path, markers: Iterable[str]) -> Path:
    """The outermost directory between ``file_path`` and the workspace root holding a root marker.

    Falls back to the workspace root, so a server never runs outside the workspace.
    """
    root = workspace_root.resolve()
    markers = tuple(markers)
    if not markers:
        return root
    found: Path | None = None
    current = file_path.resolve().parent
    while current == root or root in current.parents:
        if any((current / marker).exists() for marker in markers):
            found = current
        if current == root:
            break
        current = current.parent
    return found or root


# ------------------------------------------------------------------------ JSON-RPC connection
class RpcReplyError(Exception):
    """Raised by a server-request handler to answer with a JSON-RPC error."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ConnectionExit:
    reason: str  # eof | protocol | error | closed
    detail: str
    returncode: int | None
    stderr_tail: str


RequestHandler = Callable[[str, Any], Any]
NotificationHandler = Callable[[str, Any], None]
ConnectionCallback = Callable[["RpcConnection"], None]


class RpcConnection:
    """JSON-RPC 2.0 over a child process's stdio with LSP framing.

    The child runs in its own session/process group. When the stream ends or breaks, the process
    group is killed, pending requests fail and ``on_lost`` fires once.
    """

    def __init__(
        self,
        name: str,
        argv: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        on_request: RequestHandler,
        on_notification: NotificationHandler,
        on_lost: ConnectionCallback | None = None,
        on_exit: ConnectionCallback | None = None,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
    ):
        self.name = name
        self.argv = list(argv)
        self.cwd = cwd
        self.env = dict(env)
        self.on_request = on_request
        self.on_notification = on_notification
        self.on_lost = on_lost
        self.on_exit = on_exit
        self.max_message_bytes = max_message_bytes
        self.proc: asyncio.subprocess.Process | None = None
        self.row_id: str | None = None
        self.closing = False
        self.crash_accounted = False
        self.exit: ConnectionExit | None = None
        self._lost = False
        self._exit_reported = False
        self._pending: dict[int, tuple[str, asyncio.Future[Any]]] = {}
        self._next_id = 0
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._reply_tasks: set[asyncio.Task[None]] = set()
        self._write_lock = asyncio.Lock()
        self._stderr: deque[str] = deque(maxlen=STDERR_LINES)
        self.stderr_bytes = 0

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    @property
    def alive(self) -> bool:
        return self.proc is not None and not self._lost and self.proc.returncode is None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def start(self) -> None:
        if not self.cwd.is_dir():
            raise LSPError(
                f"cannot start language server '{self.name}': directory {self.cwd} does not exist",
                error_class=LSP_UNAVAILABLE,
            )
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.cwd),
                env=self.env,
                start_new_session=True,
                limit=1 << 20,
            )
        except FileNotFoundError as exc:
            raise LSPError(
                f"cannot start language server '{self.name}': command not found: {self.argv[0]}",
                error_class=LSP_UNAVAILABLE,
            ) from exc
        except OSError as exc:
            raise LSPError(
                f"cannot start language server '{self.name}': {exc}", error_class=LSP_UNAVAILABLE
            ) from exc
        self._reader_task = asyncio.create_task(self._read_loop(), name=f"lsp-read-{self.name}")
        self._stderr_task = asyncio.create_task(self._drain_stderr(), name=f"lsp-stderr-{self.name}")

    # ------------------------------------------------------------------ reading
    async def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        stream = self.proc.stderr
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        partial = ""
        while True:
            try:
                chunk = await stream.read(8192)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            self.stderr_bytes += len(chunk)
            *lines, partial = (partial + decoder.decode(chunk)).split("\n")
            for line in lines:
                self._stderr.append(line.rstrip("\r")[:STDERR_LINE_CHARS])
            if len(partial) > STDERR_LINE_CHARS * 4:
                self._stderr.append(partial[:STDERR_LINE_CHARS])
                partial = ""
        partial += decoder.decode(b"", final=True)
        if partial.strip():
            self._stderr.append(partial[:STDERR_LINE_CHARS])

    def _noise(self, text: str) -> None:
        self._stderr.append(f"[stdout] {text[:STDERR_LINE_CHARS]}")

    def stderr_tail(self, lines: int = 8) -> str:
        return "\n".join(list(self._stderr)[-lines:])

    async def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        reason, detail = "eof", "the server closed its output"
        try:
            while True:
                msg = await read_message(
                    self.proc.stdout, max_bytes=self.max_message_bytes, noise=self._noise
                )
                if msg is None:
                    break
                self._dispatch(msg)
        except LSPError as exc:
            reason, detail = "protocol", exc.message
        except asyncio.CancelledError:
            reason, detail = "closed", "the connection was closed"
            raise
        except Exception as exc:
            log.exception("LSP reader for %s failed", self.name)
            reason, detail = "error", f"{type(exc).__name__}: {exc}"
        finally:
            await self._connection_lost(reason, detail)

    def _dispatch(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        method = msg.get("method")
        if isinstance(method, str):
            if "id" in msg:
                task = asyncio.create_task(self._reply(msg["id"], method, msg.get("params")))
                self._reply_tasks.add(task)
                task.add_done_callback(self._reply_tasks.discard)
            else:
                try:
                    self.on_notification(method, msg.get("params"))
                except Exception:
                    log.exception("LSP notification handler for %s failed", method)
            return
        request_id = msg.get("id")
        entry = self._pending.pop(request_id, None) if isinstance(request_id, int) else None
        if entry is None:
            return  # late answer to a request that already timed out or was cancelled
        method_name, fut = entry
        if fut.done():
            return
        error = msg.get("error")
        if isinstance(error, dict):
            fut.set_exception(_rpc_error(self.name, method_name, error))
        else:
            fut.set_result(msg.get("result"))

    async def _reply(self, request_id: Any, method: str, params: Any) -> None:
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        try:
            reply["result"] = self.on_request(method, params)
        except RpcReplyError as exc:
            reply["error"] = {"code": exc.code, "message": exc.message}
        except Exception as exc:
            log.exception("LSP request handler for %s failed", method)
            reply["error"] = {"code": -32603, "message": f"client error: {type(exc).__name__}"}
        with contextlib.suppress(LSPError):
            await self._send(reply)

    async def _connection_lost(self, reason: str, detail: str) -> None:
        self._lost = True
        proc = self.proc
        if proc is not None and not self.closing and reason != "closed":
            if proc.returncode is None and reason == "eof":
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), 1.0)
            if proc.returncode is None or kill_group(proc.pid, 0):
                # Still running without usable output, or helpers left in its group: unusable either way.
                kill_group(proc.pid, signal.SIGKILL)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), 2.0)
            if self._stderr_task is not None and not self._stderr_task.done():
                await asyncio.wait({self._stderr_task}, timeout=0.5)
        self.exit = ConnectionExit(reason, detail, proc.returncode if proc else None, self.stderr_tail(5))
        for _, fut in self._pending.values():
            if not fut.done():
                fut.set_exception(self.lost_error())
        self._pending.clear()
        if self.on_lost is not None:
            try:
                self.on_lost(self)
            except Exception:
                log.exception("LSP connection-lost handler failed")
        if not self.closing and reason != "closed":
            self._report_exit()

    def lost_error(self) -> LSPError:
        info = self.exit
        if self.closing or (info is not None and info.reason == "closed"):
            return LSPError(f"language server '{self.name}' was shut down", error_class=LSP_UNAVAILABLE)
        if info is None:
            code = self.proc.returncode if self.proc is not None else None
            if code is None:
                return LSPError(f"language server '{self.name}' is not running", error_class=LSP_UNAVAILABLE)
            # The process is gone but its reader has not seen end-of-stream yet.
            info = ConnectionExit("eof", "the server exited", code, self.stderr_tail(5))
        tail = one_line(info.stderr_tail, 400)
        stderr = f"; stderr: {tail}" if tail else ""
        if info.reason == "protocol":
            return LSPError(
                f"language server '{self.name}' broke the protocol ({info.detail}) and was stopped{stderr}",
                error_class=LSP_PROTOCOL,
                hint=FALLBACK_HINT,
            )
        code = info.returncode
        if code is None:
            how = "stopped responding"
        elif code < 0:
            how = f"was killed by signal {-code}"
        else:
            how = f"exited with code {code}"
        return LSPError(
            f"language server '{self.name}' {how} unexpectedly{stderr}",
            error_class=LSP_CRASHED,
            hint=FALLBACK_HINT,
        )

    def _report_exit(self) -> None:
        if self._exit_reported or self.on_exit is None:
            return
        self._exit_reported = True
        try:
            self.on_exit(self)
        except Exception:
            log.exception("LSP process-exit handler failed")

    # ------------------------------------------------------------------ writing
    async def _send(self, msg: Mapping[str, Any]) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None or not self.alive:
            raise self.lost_error()
        data = encode_message(msg)
        async with self._write_lock:
            try:
                proc.stdin.write(data)
                await proc.stdin.drain()
            except (OSError, RuntimeError) as exc:
                raise LSPError(
                    f"language server '{self.name}' stopped accepting input: {exc}", error_class=LSP_CRASHED
                ) from exc

    def _cancel_nowait(self, request_id: int) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None or not self.alive:
            return
        with contextlib.suppress(OSError, RuntimeError):
            proc.stdin.write(
                encode_message({"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": request_id}})
            )

    async def request(self, method: str, params: Any = None, *, timeout: float) -> Any:
        self._next_id += 1
        request_id = self._next_id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = (method, fut)
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            msg["params"] = params
        try:
            await self._send(msg)
            return await asyncio.wait_for(fut, timeout)
        except TimeoutError:
            self._cancel_nowait(request_id)
            raise LSPError(
                f"language server '{self.name}' did not answer {method} within {timeout:g}s",
                error_class=LSP_TIMEOUT,
                hint=FALLBACK_HINT,
            ) from None
        except asyncio.CancelledError:
            self._cancel_nowait(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Any = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    # ------------------------------------------------------------------ closing
    def kill_now(self) -> None:
        """Synchronously SIGKILL the whole process group (safe inside cancellation paths)."""
        self.closing = True
        if self.proc is not None:
            kill_group(self.proc.pid, signal.SIGKILL)

    async def close(self, *, exit_wait: float, kill_grace: float) -> None:
        """Close stdin, wait for exit, then SIGTERM and SIGKILL the process group: no orphans."""
        self.closing = True
        proc = self.proc
        if proc is None:
            return
        if proc.stdin is not None and not proc.stdin.is_closing():
            with contextlib.suppress(OSError, RuntimeError):
                proc.stdin.close()
        if proc.returncode is None and exit_wait > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), exit_wait)
        pgid = proc.pid
        if proc.returncode is None:
            kill_group(pgid, signal.SIGTERM)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), kill_grace)
        if kill_group(pgid, 0):
            # The server ignored SIGTERM, or helpers it started are still alive in its group.
            kill_group(pgid, signal.SIGKILL)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), max(kill_grace, 1.0))
        tasks = [
            t
            for t in (self._reader_task, self._stderr_task, *self._reply_tasks)
            if t is not None and not t.done()
        ]
        if tasks:
            _, still_running = await asyncio.wait(tasks, timeout=1.0)
            for task in still_running:
                task.cancel()
            if still_running:
                await asyncio.wait(still_running, timeout=1.0)
        self._report_exit()


# ------------------------------------------------------------------------- server specs
@dataclass
class LSPSettings:
    """Timeouts and limits. Defaults suit real servers; tests shorten them."""

    request_timeout_s: float = 30.0
    startup_timeout_s: float = 60.0
    shutdown_timeout_s: float = 3.0
    exit_wait_s: float = 2.0
    kill_grace_s: float = 3.0
    max_restarts: int = 3
    restart_backoff_s: float = 1.0
    max_open_documents: int = 64
    max_clients: int = 8
    idle_timeout_s: float = 900.0


@dataclass
class ServerSpec:
    name: str
    command: tuple[str, ...]
    languages: tuple[str, ...]
    root_markers: tuple[str, ...] = ()
    source: str = "config"  # config | builtin
    init_options: dict[str, Any] | None = None
    # Merged over init_options when the project is not trusted (features that run project code).
    untrusted_init_options: dict[str, Any] | None = None
    env: dict[str, str] = field(default_factory=dict)
    offline_env: dict[str, str] = field(default_factory=dict)


_PY_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "pyrightconfig.json", "requirements.txt", "Pipfile")
BUILTIN_SERVERS: tuple[ServerSpec, ...] = (
    ServerSpec("pyright", ("pyright-langserver", "--stdio"), ("python",), _PY_MARKERS, "builtin"),
    ServerSpec("basedpyright", ("basedpyright-langserver", "--stdio"), ("python",), _PY_MARKERS, "builtin"),
    ServerSpec("pylsp", ("pylsp",), ("python",), _PY_MARKERS, "builtin"),
    ServerSpec(
        "typescript-language-server",
        ("typescript-language-server", "--stdio"),
        ("typescript", "javascript"),
        ("tsconfig.json", "jsconfig.json", "package.json"),
        "builtin",
        # Automatic type acquisition installs @types packages from npm behind the policy engine's back.
        init_options={"disableAutomaticTypingAcquisition": True},
    ),
    ServerSpec(
        "gopls",
        ("gopls",),
        ("go",),
        ("go.work", "go.mod"),
        "builtin",
        # Never download and run a toolchain named by the repository's go.mod.
        env={"GOTOOLCHAIN": "local"},
        offline_env={"GOPROXY": "off"},
    ),
    ServerSpec(
        "rust-analyzer",
        ("rust-analyzer",),
        ("rust",),
        ("Cargo.toml",),
        "builtin",
        untrusted_init_options={
            "cargo": {"buildScripts": {"enable": False}},
            "procMacro": {"enable": False},
            "checkOnSave": False,
        },
        offline_env={"CARGO_NET_OFFLINE": "true"},
    ),
    ServerSpec(
        "clangd",
        ("clangd",),
        ("c", "cpp"),
        ("compile_commands.json", "compile_flags.txt", ".clangd"),
        "builtin",
    ),
)


# ------------------------------------------------------------------------ per-root client
@dataclass
class OpenDocument:
    uri: str
    path: Path
    rel: str
    language_id: str
    version: int
    digest: str
    lines: list[str]
    synced_seq: int


@dataclass
class Location:
    uri: str
    line: int  # 0-based
    character: int  # in the server's position encoding


@dataclass
class DiagnosticsResult:
    items: list[dict[str, Any]]
    mode: str  # pull | push
    fresh: bool
    received: bool


@dataclass
class _PushEntry:
    seq: int
    version: int | None
    items: list[dict[str, Any]]


@dataclass
class _PullEntry:
    result_id: str | None
    digest: str
    items: list[dict[str, Any]]


@dataclass
class ClientHooks:
    """Runtime integration points for a client (process bookkeeping and events)."""

    process_started: Callable[[RpcConnection], str | None]
    process_ended: Callable[[RpcConnection], None]
    event: Callable[[str, str, dict[str, Any]], None]


def _read_source(path: Path, rel: str) -> bytes:
    try:
        size = path.stat().st_size
        if size > MAX_SYNC_BYTES:
            raise LSPError(
                f"{rel} is {size} bytes; language server sync is limited to {MAX_SYNC_BYTES} bytes",
                error_class=LSP_FILE_TOO_LARGE,
                hint=FALLBACK_HINT,
            )
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ToolError(f"{rel} does not exist", error_class="not_found") from exc
    except IsADirectoryError as exc:
        raise ToolError(f"{rel} is a directory", error_class="not_a_file") from exc
    if b"\x00" in data[:8192]:
        raise LSPError(f"{rel} is a binary file", error_class=LSP_UNSUPPORTED)
    return data


def _supported(value: Any) -> bool:
    # Server capabilities are `true` or an options object; `{}` means supported.
    return value is True or isinstance(value, dict)


class LanguageServerClient:
    """One language server process for one root, restarted after crashes within a budget."""

    def __init__(
        self,
        spec: ServerSpec,
        root: Path,
        *,
        argv: list[str],
        env: Mapping[str, str],
        settings: LSPSettings,
        init_options: dict[str, Any] | None = None,
        hooks: ClientHooks | None = None,
    ):
        self.spec = spec
        self.root = root.resolve()
        self.root_uri = self.root.as_uri()
        self.argv = list(argv)
        self.env = dict(env)
        self.settings = settings
        self.init_options = init_options
        self.hooks = hooks
        self.state = "idle"  # idle | starting | ready | crashed | failed | closed
        self.conn: RpcConnection | None = None
        self.capabilities: dict[str, Any] = {}
        self.server_info: dict[str, Any] = {}
        self.encoding = "utf-16"
        self.starts = 0
        self.crashes = 0
        self.restarts = 0
        self.last_error: str | None = None
        self.failure: LSPError | None = None
        self.last_used = time.monotonic()
        self.busy = 0
        self._started_at: float | None = None
        self._last_crash_at: float | None = None
        self._start_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
        self._docs: OrderedDict[str, OpenDocument] = OrderedDict()
        self._push: dict[str, _PushEntry] = {}
        self._pull: dict[str, _PullEntry] = {}
        self._pull_unsupported = False
        self._diag_seq = 0
        self._diag_event = asyncio.Event()
        self._registrations: dict[str, str] = {}
        self._progress: set[str] = set()
        self.messages: deque[str] = deque(maxlen=20)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def pid(self) -> int | None:
        return self.conn.pid if self.conn is not None and self.conn.alive else None

    # ------------------------------------------------------------------ lifecycle
    async def ensure_ready(self) -> RpcConnection:
        async with self._start_lock:
            conn = self.conn
            if self.state == "ready" and conn is not None and conn.alive:
                return conn
            if self.state == "closed":
                raise LSPError(
                    f"language server '{self.name}' has been shut down", error_class=LSP_UNAVAILABLE
                )
            if self.failure is not None:
                raise self.failure.copy()
            if self.state == "ready" and conn is not None and not conn.crash_accounted:
                # Died before its reader saw end-of-stream: account for it like any other crash.
                self._crashed(conn)
            if self.state == "crashed":
                if self.restarts >= self.settings.max_restarts:
                    self._fail(
                        LSPError(
                            f"language server '{self.name}' crashed {self.crashes} time(s) and its restart budget "
                            f"({self.settings.max_restarts}) is exhausted; last error: {self.last_error}",
                            error_class=LSP_CRASHED,
                            hint=FALLBACK_HINT,
                        )
                    )
                    assert self.failure is not None
                    raise self.failure.copy()
                backoff = min(self.settings.restart_backoff_s * (2**self.restarts), MAX_BACKOFF_S)
                delay = backoff - (time.monotonic() - (self._last_crash_at or 0.0))
                if delay > 0:
                    await asyncio.sleep(delay)
                self.restarts += 1
            await self._start()
            assert self.conn is not None
            return self.conn

    def _reset_session(self) -> None:
        self._docs.clear()
        self._push.clear()
        self._pull.clear()
        self._pull_unsupported = False
        self._registrations.clear()
        self._progress.clear()
        self.capabilities = {}
        self.server_info = {}
        self.encoding = "utf-16"

    async def _start(self) -> None:
        self.state = "starting"
        self._reset_session()
        conn = RpcConnection(
            self.name,
            self.argv,
            cwd=self.root,
            env=self.env,
            on_request=self._on_server_request,
            on_notification=self._on_notification,
            on_lost=self._on_lost,
            on_exit=self._on_exit,
        )
        self.conn = conn
        try:
            await conn.start()
        except LSPError as exc:
            if self.state != "closed":
                self._fail(exc)  # the executable cannot be started; retrying will not help
            raise
        self.starts += 1
        self._started_at = time.monotonic()
        if self.hooks is not None:
            conn.row_id = self.hooks.process_started(conn)
        try:
            self._raise_if_closed()
            result = await conn.request(
                "initialize", self._initialize_params(), timeout=self.settings.startup_timeout_s
            )
            if not isinstance(result, dict):
                raise LSPError(
                    f"language server '{self.name}' answered initialize without a result object",
                    error_class=LSP_PROTOCOL,
                )
            caps = result.get("capabilities")
            self.capabilities = caps if isinstance(caps, dict) else {}
            info = result.get("serverInfo")
            self.server_info = info if isinstance(info, dict) else {}
            encoding = self.capabilities.get("positionEncoding")
            self.encoding = encoding if encoding in ("utf-8", "utf-16", "utf-32") else "utf-16"
            await conn.notify("initialized", {})
            self._raise_if_closed()
        except BaseException as exc:
            await self._abort_start(conn, exc)
            raise
        self.state = "ready"
        self._emit(
            "lsp.started",
            "info",
            {
                "pid": conn.pid,
                "server_info": self.server_info,
                "restarts": self.restarts,
                "encoding": self.encoding,
            },
        )

    async def _abort_start(self, conn: RpcConnection, exc: BaseException) -> None:
        # The failed attempt's process must never outlive it, even when we are being cancelled.
        conn.kill_now()
        if self.state == "starting":
            if isinstance(exc, LSPError):
                message = exc.message
                if exc.error_class == LSP_TIMEOUT:
                    message = f"language server '{self.name}' did not initialize within {self.settings.startup_timeout_s:g}s"
                self._record_crash(conn, message)
                self._emit("lsp.start_failed", "warning", {"error": message, "error_class": exc.error_class})
            else:
                self.state = "idle"
        await conn.close(exit_wait=0, kill_grace=1.0)

    def _raise_if_closed(self) -> None:
        if self.state == "closed":
            raise LSPError(
                f"language server '{self.name}' was shut down during startup", error_class=LSP_UNAVAILABLE
            )

    def _record_crash(self, conn: RpcConnection, message: str) -> None:
        conn.crash_accounted = True
        now = time.monotonic()
        if self._started_at is not None and now - self._started_at >= STABLE_UPTIME_S:
            self.restarts = 0  # a long healthy run earns back the restart budget
        self.crashes += 1
        self._last_crash_at = now
        self.last_error = message
        self.state = "crashed"

    def _crashed(self, conn: RpcConnection) -> None:
        error = conn.lost_error()
        self._record_crash(conn, error.message)
        exit_info = conn.exit
        self._emit(
            "lsp.crashed",
            "warning",
            {
                "pid": conn.pid,
                "error_class": error.error_class,
                "returncode": conn.proc.returncode if conn.proc is not None else None,
                "stderr": exit_info.stderr_tail[-800:] if exit_info else conn.stderr_tail(5)[-800:],
            },
        )

    def _fail(self, exc: LSPError) -> None:
        self.failure = exc
        self.last_error = exc.message
        self.state = "failed"
        self._emit("lsp.failed", "error", {"error": exc.message, "error_class": exc.error_class})

    def _on_lost(self, conn: RpcConnection) -> None:
        self._wake_diagnostics()
        if (
            conn is not self.conn
            or conn.closing
            or conn.crash_accounted
            or self.state in ("closed", "starting")
        ):
            return
        self._crashed(conn)

    def _on_exit(self, conn: RpcConnection) -> None:
        if self.hooks is not None:
            self.hooks.process_ended(conn)

    def _emit(self, kind: str, level: str, data: dict[str, Any]) -> None:
        if self.hooks is not None:
            self.hooks.event(kind, level, {"server": self.name, "root": str(self.root), **data})

    def _initialize_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "processId": os.getpid(),
            "clientInfo": {"name": "core-main", "version": __version__},
            "locale": "en",
            "rootPath": str(self.root),
            "rootUri": self.root_uri,
            "workspaceFolders": [{"uri": self.root_uri, "name": self.root.name or str(self.root)}],
            "capabilities": CLIENT_CAPABILITIES,
            "trace": "off",
        }
        if self.init_options is not None:
            params["initializationOptions"] = self.init_options
        return params

    async def aclose(self) -> None:
        previous = self.state
        self.state = "closed"
        conn = self.conn
        if conn is None or conn.proc is None:
            return
        conn.closing = True
        if previous == "ready" and conn.alive:
            with contextlib.suppress(LSPError):
                await conn.request("shutdown", None, timeout=self.settings.shutdown_timeout_s)
            with contextlib.suppress(LSPError):
                await conn.notify("exit")
        await conn.close(exit_wait=self.settings.exit_wait_s, kill_grace=self.settings.kill_grace_s)
        self._emit("lsp.stopped", "debug", {"pid": conn.pid, "returncode": conn.proc.returncode})

    def _live_conn(self) -> RpcConnection:
        conn = self.conn
        if conn is not None and conn.alive and self.state == "ready":
            return conn
        if conn is not None and conn.exit is not None:
            raise conn.lost_error()
        raise LSPError(f"language server '{self.name}' is not running", error_class=LSP_UNAVAILABLE)

    # ------------------------------------------------------------ server → client
    def _on_server_request(self, method: str, params: Any) -> Any:
        body = params if isinstance(params, dict) else {}
        if method == "workspace/configuration":
            items = body.get("items")
            return [None] * (len(items) if isinstance(items, list) else 0)
        if method == "client/registerCapability":
            for reg in body.get("registrations") or []:
                if (
                    isinstance(reg, dict)
                    and isinstance(reg.get("id"), str)
                    and isinstance(reg.get("method"), str)
                ):
                    self._registrations[reg["id"]] = reg["method"]
            return None
        if method == "client/unregisterCapability":
            # The field name really is misspelled in the protocol.
            for reg in body.get("unregisterations") or []:
                if isinstance(reg, dict):
                    self._registrations.pop(str(reg.get("id")), None)
            return None
        if method == "window/workDoneProgress/create":
            return None
        if method == "workspace/workspaceFolders":
            return [{"uri": self.root_uri, "name": self.root.name or str(self.root)}]
        if method == "workspace/applyEdit":
            return {
                "applied": False,
                "failureReason": "Core Main's language-server integration is read-only; edits go through Core Main's own tools",
            }
        if method == "window/showMessageRequest":
            self._remember_message(body)
            return None
        if method == "window/showDocument":
            return {"success": False}
        if method.startswith("workspace/") and method.endswith("/refresh"):
            if method == "workspace/diagnostic/refresh":
                self._pull.clear()
            return None
        raise RpcReplyError(-32601, f"Core Main does not handle {method}")

    def _on_notification(self, method: str, params: Any) -> None:
        body = params if isinstance(params, dict) else {}
        if method == "textDocument/publishDiagnostics":
            uri, items = body.get("uri"), body.get("diagnostics")
            if not isinstance(uri, str) or not isinstance(items, list):
                return
            self._diag_seq += 1
            version = body.get("version")
            key = normalize_uri(uri)
            self._push.pop(key, None)
            self._push[key] = _PushEntry(
                self._diag_seq,
                version if isinstance(version, int) else None,
                [d for d in items if isinstance(d, dict)][:MAX_STORED_DIAGNOSTICS],
            )
            while len(self._push) > MAX_DIAGNOSTIC_URIS:
                self._push.pop(next(iter(self._push)))
            self._wake_diagnostics()
        elif method in ("window/logMessage", "window/showMessage"):
            self._remember_message(body)
        elif method == "$/progress":
            token, value = body.get("token"), body.get("value")
            if isinstance(value, dict) and isinstance(token, (str, int)):
                if value.get("kind") == "begin":
                    self._progress.add(str(token))
                elif value.get("kind") == "end":
                    self._progress.discard(str(token))

    def _remember_message(self, body: Mapping[str, Any]) -> None:
        kind = {1: "error", 2: "warning", 3: "info", 4: "log"}.get(body.get("type") or 0, "log")
        message = body.get("message")
        if isinstance(message, str) and kind in ("error", "warning"):
            self.messages.append(f"{kind}: {one_line(message, 300)}")

    def _wake_diagnostics(self) -> None:
        event, self._diag_event = self._diag_event, asyncio.Event()
        event.set()

    # ------------------------------------------------------------------ documents
    def _sync_mode(self) -> tuple[bool, int]:
        sync = self.capabilities.get("textDocumentSync")
        if isinstance(sync, bool):
            return False, 0
        if isinstance(sync, int):
            return sync != 0, sync
        if isinstance(sync, dict):
            change = sync.get("change")
            return bool(sync.get("openClose")), change if isinstance(change, int) else 0
        return False, 0

    async def sync(self, path: Path, rel: str, language: str) -> OpenDocument:
        """Make the server's view of ``path`` match the file on disk; returns the open document."""
        data = await asyncio.to_thread(_read_source, path, rel)
        await self.ensure_ready()
        digest = sha256_hex(data)
        uri = path.as_uri()
        async with self._sync_lock:
            conn = self._live_conn()
            doc = self._docs.get(uri)
            if doc is not None and doc.digest == digest:
                self._docs.move_to_end(uri)
                return doc
            text = data.decode("utf-8", errors="replace")
            lines = split_lines(text)
            open_close, change = self._sync_mode()
            if doc is None:
                while len(self._docs) >= max(1, self.settings.max_open_documents):
                    old_uri, _ = self._docs.popitem(last=False)
                    self._push.pop(old_uri, None)
                    self._pull.pop(old_uri, None)
                    if open_close:
                        await conn.notify("textDocument/didClose", {"textDocument": {"uri": old_uri}})
                doc = OpenDocument(
                    uri, path, rel, language_id(rel, language), 1, digest, lines, self._diag_seq
                )
                if open_close:
                    await conn.notify(
                        "textDocument/didOpen", {"textDocument": self._text_document_item(doc, text)}
                    )
            else:
                doc.version += 1
                doc.digest, doc.lines, doc.synced_seq = digest, lines, self._diag_seq
                if open_close and change == 0:
                    await conn.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
                    await conn.notify(
                        "textDocument/didOpen", {"textDocument": self._text_document_item(doc, text)}
                    )
                elif open_close:
                    await conn.notify(
                        "textDocument/didChange",
                        {
                            "textDocument": {"uri": uri, "version": doc.version},
                            "contentChanges": [{"text": text}],
                        },
                    )
            self._docs[uri] = doc
            self._docs.move_to_end(uri)
            return doc

    @staticmethod
    def _text_document_item(doc: OpenDocument, text: str) -> dict[str, Any]:
        return {"uri": doc.uri, "languageId": doc.language_id, "version": doc.version, "text": text}

    def open_document_lines(self) -> dict[str, list[str]]:
        return {uri: doc.lines for uri, doc in self._docs.items()}

    def position(self, doc: OpenDocument, line: int, column: int) -> dict[str, int]:
        text = doc.lines[line] if 0 <= line < len(doc.lines) else ""
        return {"line": line, "character": to_lsp_character(text, column, self.encoding)}

    # ------------------------------------------------------------------ queries
    async def _request(self, method: str, params: Any, *, capability: str | None) -> Any:
        conn = self._live_conn()
        if capability is not None and not (
            _supported(self.capabilities.get(capability)) or method in self._registrations.values()
        ):
            raise LSPError(
                f"language server '{self.name}' does not support {method}",
                error_class=LSP_UNSUPPORTED,
                hint=FALLBACK_HINT,
            )
        try:
            return await conn.request(method, params, timeout=self.settings.request_timeout_s)
        except LSPError as exc:
            if exc.rpc_code not in _RETRYABLE_RPC_CODES:
                raise
        # The server dropped the request because its state moved on; one retry is enough.
        await asyncio.sleep(0.05)
        return await self._live_conn().request(method, params, timeout=self.settings.request_timeout_s)

    async def definition(self, doc: OpenDocument, line: int, column: int) -> list[Location]:
        params = {"textDocument": {"uri": doc.uri}, "position": self.position(doc, line, column)}
        return parse_locations(
            await self._request("textDocument/definition", params, capability="definitionProvider")
        )

    async def references(
        self, doc: OpenDocument, line: int, column: int, *, include_declaration: bool = True
    ) -> list[Location]:
        params = {
            "textDocument": {"uri": doc.uri},
            "position": self.position(doc, line, column),
            "context": {"includeDeclaration": include_declaration},
        }
        return parse_locations(
            await self._request("textDocument/references", params, capability="referencesProvider")
        )

    async def hover(self, doc: OpenDocument, line: int, column: int) -> str:
        params = {"textDocument": {"uri": doc.uri}, "position": self.position(doc, line, column)}
        result = await self._request("textDocument/hover", params, capability="hoverProvider")
        return hover_text(result.get("contents")) if isinstance(result, dict) else ""

    async def document_symbols(self, doc: OpenDocument) -> list[dict[str, Any]]:
        result = await self._request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": doc.uri}},
            capability="documentSymbolProvider",
        )
        return [s for s in result if isinstance(s, dict)] if isinstance(result, list) else []

    def _supports_pull(self) -> bool:
        return not self._pull_unsupported and (
            _supported(self.capabilities.get("diagnosticProvider"))
            or "textDocument/diagnostic" in self._registrations.values()
        )

    async def diagnostics(self, doc: OpenDocument, *, wait_s: float) -> DiagnosticsResult:
        if self._supports_pull():
            try:
                return await self._pull_diagnostics(doc)
            except LSPError as exc:
                if exc.error_class != LSP_UNSUPPORTED:
                    raise
                self._pull_unsupported = True
        return await self._push_diagnostics(doc, wait_s)

    async def _pull_diagnostics(self, doc: OpenDocument) -> DiagnosticsResult:
        params: dict[str, Any] = {"textDocument": {"uri": doc.uri}}
        provider = self.capabilities.get("diagnosticProvider")
        if isinstance(provider, dict) and isinstance(provider.get("identifier"), str):
            params["identifier"] = provider["identifier"]
        previous = self._pull.get(doc.uri)
        if previous is not None and previous.result_id and previous.digest == doc.digest:
            params["previousResultId"] = previous.result_id
        report = await self._request("textDocument/diagnostic", params, capability=None)
        if not isinstance(report, dict):
            raise LSPError(
                f"language server '{self.name}' returned no diagnostic report", error_class=LSP_PROTOCOL
            )
        kind = report.get("kind")
        result_id = report.get("resultId") if isinstance(report.get("resultId"), str) else None
        if kind == "unchanged":
            if previous is None:
                raise LSPError(
                    f"language server '{self.name}' reported unchanged diagnostics without a previous report",
                    error_class=LSP_PROTOCOL,
                )
            items = previous.items
        elif kind == "full":
            items = [d for d in report.get("items") or [] if isinstance(d, dict)][:MAX_STORED_DIAGNOSTICS]
        else:
            raise LSPError(
                f"language server '{self.name}' returned an unknown diagnostic report kind {kind!r}",
                error_class=LSP_PROTOCOL,
            )
        self._pull[doc.uri] = _PullEntry(result_id, doc.digest, items)
        return DiagnosticsResult(items, "pull", fresh=True, received=True)

    async def _push_diagnostics(self, doc: OpenDocument, wait_s: float) -> DiagnosticsResult:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, wait_s)
        while True:
            entry = self._push.get(doc.uri)
            if (
                entry is not None
                and entry.seq > doc.synced_seq
                and (entry.version is None or entry.version >= doc.version)
            ):
                return DiagnosticsResult(entry.items, "push", fresh=True, received=True)
            self._live_conn()
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._diag_event.wait(), remaining)
        if entry is not None:
            return DiagnosticsResult(entry.items, "push", fresh=False, received=True)
        return DiagnosticsResult([], "push", fresh=False, received=False)

    def describe(self) -> dict[str, Any]:
        conn = self.conn
        return {
            "root": str(self.root),
            "state": self.state,
            "pid": self.pid,
            "server_info": self.server_info,
            "position_encoding": self.encoding,
            "starts": self.starts,
            "crashes": self.crashes,
            "restarts": self.restarts,
            "max_restarts": self.settings.max_restarts,
            "last_error": self.last_error,
            "open_documents": len(self._docs),
            "pending_requests": conn.pending_count if conn is not None else 0,
            "indexing": bool(self._progress),
            "recent_messages": list(self.messages)[-5:],
            "stderr_tail": conn.stderr_tail(5) if conn is not None else "",
        }


# ---------------------------------------------------------------------------- results
def parse_locations(result: Any) -> list[Location]:
    """Normalize ``Location | Location[] | LocationLink[] | null`` into de-duplicated locations."""
    items = result if isinstance(result, list) else [result] if isinstance(result, dict) else []
    seen: set[tuple[str, int, int]] = set()
    out: list[Location] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "targetUri" in item:
            uri, rng = item.get("targetUri"), item.get("targetSelectionRange") or item.get("targetRange")
        else:
            uri, rng = item.get("uri"), item.get("range")
        start = rng.get("start") if isinstance(rng, dict) else None
        if not isinstance(uri, str) or not isinstance(start, dict):
            continue
        line, character = start.get("line"), start.get("character")
        if not isinstance(line, int) or not isinstance(character, int) or line < 0 or character < 0:
            continue
        key = (normalize_uri(uri), line, character)
        if key not in seen:
            seen.add(key)
            out.append(Location(*key))
    return out


def hover_text(contents: Any) -> str:
    """Flatten ``MarkupContent | MarkedString | MarkedString[]`` to text."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        value = str(contents.get("value") or "")
        language = contents.get("language")
        if language and "kind" not in contents:
            return f"```{language}\n{value}\n```"
        return value
    if isinstance(contents, list):
        return "\n\n".join(part for part in (hover_text(c) for c in contents) if part.strip())
    return ""


class _Sources:
    """Line access for rendering results: open documents first, then workspace files on disk.

    Files outside the workspace, inside ``.git`` or matching sensitive patterns are never read.
    """

    def __init__(self, root: Path, open_docs: Mapping[str, list[str]], sensitive: Iterable[str]):
        self.root = root.resolve()
        self.open_docs = open_docs
        self.sensitive = tuple(sensitive)
        self._cache: dict[Path, list[str] | None] = {}

    def _lines(self, uri: str, path: Path) -> list[str] | None:
        if uri in self.open_docs:
            return self.open_docs[uri]
        if path not in self._cache:
            lines: list[str] | None = None
            try:
                if path.is_file() and path.stat().st_size <= MAX_SYNC_BYTES:
                    data = path.read_bytes()
                    if b"\x00" not in data[:8192]:
                        lines = split_lines(data.decode("utf-8", errors="replace"))
            except OSError:
                lines = None
            self._cache[path] = lines
        return self._cache[path]

    def describe(self, loc: Location, encoding: str) -> dict[str, Any]:
        out: dict[str, Any] = {"path": loc.uri, "line": loc.line + 1, "column": loc.character + 1}
        path = uri_to_path(loc.uri)
        if path is None:
            out["note"] = "non-file location"
            return out
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            resolved = path
        if resolved != self.root and self.root not in resolved.parents:
            out.update(path=str(path), note="outside the workspace; not read")
            return out
        rel = resolved.relative_to(self.root).as_posix()
        out["path"] = rel
        if rel.split("/", 1)[0] == ".git" or sensitive_reason(rel, self.sensitive):
            out["note"] = "sensitive file; preview withheld"
            return out
        lines = self._lines(loc.uri, resolved)
        if lines is not None and loc.line < len(lines):
            text = lines[loc.line]
            out["column"] = from_lsp_character(text, loc.character, encoding) + 1
            preview = one_line(text.strip(), PREVIEW_CHARS)
            if preview:
                out["preview"] = preview
        return out


def format_location(d: Mapping[str, Any]) -> str:
    text = f"{d['path']}:{d['line']}:{d['column']}"
    if d.get("preview"):
        text += f"  {d['preview']}"
    if d.get("note"):
        text += f"  ({d['note']})"
    return text


_IDENTIFIER = re.compile(r"[\w$]+")
_DEFINITION_KEYWORDS = (
    r"(?:def|class|function|func|fn|struct|enum|trait|interface|type|typedef|const|let|var|val|impl|"
    r"module|namespace|record|object|protocol|macro)"
)


def _check_line(lines: list[str], rel: str, line: int) -> None:
    if line > len(lines):
        raise ToolError(
            f"line {line} is past the end of {rel} ({len(lines)} lines)", error_class="invalid_position"
        )


def resolve_position(
    lines: list[str], rel: str, line: int | None, character: int | None, symbol: str | None
) -> tuple[int, int, str | None]:
    """Map tool arguments (1-based line/column or a symbol name) to a 0-based (line, column).

    Returns a note describing how a symbol was resolved (``None`` for explicit positions).
    """
    if symbol is None:
        if line is None or character is None:
            raise ToolError(
                "give line and character (both 1-based), or a symbol", error_class="invalid_arguments"
            )
        _check_line(lines, rel, line)
        text = lines[line - 1]
        if character > len(text) + 1:
            raise ToolError(
                f"column {character} is past the end of {rel}:{line} ({len(text)} characters)",
                error_class="invalid_position",
            )
        return line - 1, character - 1, None
    name = symbol.strip().rstrip("()").rsplit(".", 1)[-1].rsplit("::", 1)[-1].strip()
    if not name:
        raise ToolError(f"cannot locate symbol {symbol!r}", error_class="invalid_arguments")
    word = re.compile(rf"(?<![\w$]){re.escape(name)}(?![\w$])")
    if line is not None:
        _check_line(lines, rel, line)
        starts = [m.start() for m in word.finditer(lines[line - 1])]
        if not starts:
            raise ToolError(f"'{name}' does not occur on {rel}:{line}", error_class="symbol_not_found")
        target = (character - 1) if character is not None else 0
        column = min(starts, key=lambda c: abs(c - target))
        return line - 1, column, f"'{name}' found on line {line}"
    definition = re.compile(rf"\b{_DEFINITION_KEYWORDS}\s+[*&]?{re.escape(name)}(?![\w$])")
    first: tuple[int, int] | None = None
    for index, text in enumerate(lines):
        m = definition.search(text)
        if m is not None:
            return index, m.end() - len(name), f"definition of '{name}' in {rel}"
        if first is None:
            occurrence = word.search(text)
            if occurrence is not None:
                first = (index, occurrence.start())
    if first is not None:
        return first[0], first[1], f"first occurrence of '{name}' in {rel}"
    raise ToolError(
        f"'{name}' does not occur in {rel}; locate it with find_symbol or search_text first",
        error_class="symbol_not_found",
    )


def word_at(text: str, column: int) -> str | None:
    for m in _IDENTIFIER.finditer(text):
        if m.start() <= column < m.end() or (m.end() == column and column == len(text)):
            return m.group()
    return None


def _subject(doc: OpenDocument, line: int, column: int) -> str:
    text = doc.lines[line] if 0 <= line < len(doc.lines) else ""
    word = word_at(text, column)
    where = f"{doc.rel}:{line + 1}:{column + 1}"
    return f"'{word}' at {where}" if word else where


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default


def flatten_symbols(symbols: list[dict[str, Any]], lines: list[str], encoding: str) -> list[dict[str, Any]]:
    """Hierarchical ``DocumentSymbol[]`` or flat ``SymbolInformation[]`` → indented entries."""
    out: list[dict[str, Any]] = []

    def start_of(symbol: Mapping[str, Any]) -> tuple[int, int]:
        rng = symbol.get("selectionRange") or symbol.get("range")
        if not isinstance(rng, dict):
            location = symbol.get("location")
            rng = location.get("range") if isinstance(location, dict) else None
        start = rng.get("start") if isinstance(rng, dict) else None
        if not isinstance(start, dict):
            return 0, 0
        line, character = _int(start.get("line")), _int(start.get("character"))
        text = lines[line] if 0 <= line < len(lines) else ""
        return line + 1, from_lsp_character(text, character, encoding) + 1

    def walk(items: list[Any], depth: int) -> None:
        for symbol in items:
            if not isinstance(symbol, dict) or len(out) >= MAX_SYMBOLS * 10:
                continue
            line, column = start_of(symbol)
            kind = symbol.get("kind")
            out.append(
                {
                    "name": one_line(str(symbol.get("name", "?")), 120),
                    "kind": SYMBOL_KINDS.get(kind, "symbol") if isinstance(kind, int) else "symbol",
                    "detail": one_line(str(symbol.get("detail") or ""), 120),
                    "container": one_line(str(symbol.get("containerName") or ""), 120),
                    "line": line,
                    "column": column,
                    "depth": depth,
                }
            )
            children = symbol.get("children")
            if isinstance(children, list) and depth < 20:
                walk(children, depth + 1)

    walk(symbols, 0)
    return out


def _diagnostic_entry(d: Mapping[str, Any], lines: list[str], encoding: str) -> dict[str, Any]:
    rng = d.get("range")
    start = rng.get("start") if isinstance(rng, dict) else None
    start = start if isinstance(start, dict) else {}
    line, character = _int(start.get("line")), _int(start.get("character"))
    text = lines[line] if 0 <= line < len(lines) else ""
    severity = d.get("severity")
    code = d.get("code")
    return {
        "line": line + 1,
        "column": from_lsp_character(text, character, encoding) + 1,
        "severity": SEVERITIES.get(severity, "error") if isinstance(severity, int) else "error",
        "source": str(d.get("source") or ""),
        "code": "" if code is None else str(code),
        "message": one_line(str(d.get("message") or ""), 300),
    }


# ------------------------------------------------------------------------------- manager
@dataclass
class _Candidate:
    spec: ServerSpec
    executable: str | None
    serves: tuple[str, ...]
    reason: str | None

    @property
    def available(self) -> bool:
        return self.reason is None and self.executable is not None


class LSPManager:
    """The ``lsp`` runtime extension: server discovery, per-root clients, status and tools."""

    def __init__(self, rt: CoreRuntime, *, settings: LSPSettings | None = None):
        self.rt = rt
        self.settings = settings or LSPSettings(kill_grace_s=max(0.5, float(rt.config.exec.kill_grace_s)))
        self.enabled = bool(rt.config.intel.lsp_enabled)
        self._env = build_subprocess_env(rt.env, passthrough=rt.config.permissions.env_passthrough)
        self._candidates = self._discover()
        self._clients: dict[tuple[str, str], LanguageServerClient] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._background: set[asyncio.Task[None]] = set()
        self._scanned: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------ discovery
    def _which(self, command: str) -> str | None:
        if "/" in command:
            path = Path(command).expanduser()
            if not path.is_absolute() and self.rt.project_root is not None:
                path = self.rt.project_root / path
            return str(path) if path.is_file() and os.access(path, os.X_OK) else None
        return shutil.which(command, path=self._env.get("PATH"))

    def _discover(self) -> list[_Candidate]:
        configured: list[ServerSpec] = []
        for name, cfg in self.rt.config.intel.lsp_servers.items():
            languages = tuple(dict.fromkeys(lang.strip().lower() for lang in cfg.languages if lang.strip()))
            configured.append(
                ServerSpec(name, tuple(cfg.command), languages, tuple(cfg.root_markers), "config")
            )
        covered = {lang for spec in configured for lang in spec.languages}
        out: list[_Candidate] = []
        for spec in configured:
            exe = self._which(spec.command[0]) if spec.command else None
            if not spec.command:
                reason: str | None = "empty command"
            elif not spec.languages:
                reason = "no languages configured"
            elif exe is None:
                reason = f"executable '{spec.command[0]}' not found on PATH"
            else:
                reason = None
            out.append(_Candidate(spec, exe, spec.languages, reason))
        configured_names = {spec.name for spec in configured}
        for spec in BUILTIN_SERVERS:
            if spec.name in configured_names:
                continue
            serves = tuple(lang for lang in spec.languages if lang not in covered)
            exe = self._which(spec.command[0])
            if not serves:
                reason = f"{', '.join(spec.languages)} handled by configured server(s)"
            elif exe is None:
                reason = f"not installed ('{spec.command[0]}' not found on PATH)"
            else:
                reason = None
            out.append(_Candidate(spec, exe, serves, reason))
        return out

    def server_for(self, language: str | None, rel: str) -> _Candidate:
        if not self.enabled:
            raise LSPError(
                "language servers are disabled by configuration (intel.lsp_enabled = false)",
                error_class=LSP_UNAVAILABLE,
                hint=FALLBACK_HINT,
            )
        if language is None:
            raise LSPError(
                f"cannot tell the language of {rel}, so no language server applies",
                error_class=LSP_UNAVAILABLE,
                hint=FALLBACK_HINT,
            )
        reasons: list[str] = []
        for cand in self._candidates:
            if language in cand.serves:
                if cand.available:
                    return cand
                reasons.append(f"{cand.spec.name}: {cand.reason}")
        detail = "; ".join(reasons) if reasons else "no configured or built-in server handles it"
        raise LSPError(
            f"no language server available for {language} files ({detail})",
            error_class=LSP_UNAVAILABLE,
            hint=f"{FALLBACK_HINT}, or configure a server under [intel.lsp_servers]",
        )

    def _client_env(self, spec: ServerSpec) -> dict[str, str]:
        env = dict(self._env)
        env.update(spec.env)
        if self.rt.config.permissions.network.offline:
            env.update(spec.offline_env)
        return env

    def _init_options(self, spec: ServerSpec) -> dict[str, Any] | None:
        options = dict(spec.init_options or {})
        if spec.untrusted_init_options and not self.rt.project_trusted:
            options.update(spec.untrusted_init_options)
        return options or None

    async def client_for(
        self, cand: _Candidate, workspace_root: Path, file_path: Path
    ) -> LanguageServerClient:
        root = lsp_root(workspace_root, file_path, cand.spec.root_markers)
        key = (cand.spec.name, str(root))
        async with self._lock:
            if self._closed:
                raise LSPError("the language server manager has been shut down", error_class=LSP_UNAVAILABLE)
            self._reap_stale()
            client = self._clients.get(key)
            if client is None:
                self._make_room()
                assert cand.executable is not None
                client = LanguageServerClient(
                    cand.spec,
                    root,
                    argv=[cand.executable, *cand.spec.command[1:]],
                    env=self._client_env(cand.spec),
                    settings=self.settings,
                    init_options=self._init_options(cand.spec),
                    hooks=ClientHooks(self._process_started, self._process_ended, self._event),
                )
                self._clients[key] = client
            client.last_used = time.monotonic()
            return client

    def _reap_stale(self) -> None:
        """Stop servers whose workspace disappeared or that sat idle."""
        now = time.monotonic()
        for key, client in list(self._clients.items()):
            gone = not client.root.is_dir()
            idle = (
                client.busy == 0
                and client.pid is not None
                and now - client.last_used > self.settings.idle_timeout_s
            )
            if gone or idle:
                del self._clients[key]
                self._close_in_background(client)

    def _make_room(self) -> None:
        """Before starting another server, stop least-recently-used idle ones beyond ``max_clients``."""
        live = sorted(
            (c for c in self._clients.items() if c[1].pid is not None and c[1].busy == 0),
            key=lambda item: item[1].last_used,
        )
        running = sum(1 for c in self._clients.values() if c.pid is not None)
        while live and running >= max(1, self.settings.max_clients):
            key, client = live.pop(0)
            del self._clients[key]
            self._close_in_background(client)
            running -= 1

    def _close_in_background(self, client: LanguageServerClient) -> None:
        task = asyncio.get_running_loop().create_task(client.aclose())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    @contextlib.asynccontextmanager
    async def document(
        self, ctx: ToolContext, raw_path: str
    ) -> AsyncIterator[tuple[LanguageServerClient, OpenDocument]]:
        """Resolve a workspace path, pick its server and sync the file before a query."""
        path, rel = guard(ctx, raw_path)
        if not path.exists():
            raise ToolError(f"{rel} does not exist", error_class="not_found")
        if not path.is_file():
            raise ToolError(f"{rel} is not a regular file", error_class="not_a_file")
        language = detect_language(rel)
        cand = self.server_for(language, rel)
        client = await self.client_for(cand, ctx.workspace.path, path)
        client.busy += 1
        try:
            doc = await client.sync(path, rel, language or "plaintext")
            yield client, doc
        finally:
            client.busy -= 1
            client.last_used = time.monotonic()

    async def describe_locations(
        self, ctx: ToolContext, client: LanguageServerClient, locations: list[Location]
    ) -> list[dict[str, Any]]:
        sources = _Sources(
            ctx.workspace.path, client.open_document_lines(), ctx.services.config.permissions.sensitive_paths
        )
        encoding = client.encoding
        return await asyncio.to_thread(lambda: [sources.describe(loc, encoding) for loc in locations])

    # ------------------------------------------------------------------ languages
    def _project_languages(self) -> set[str]:
        project = self.rt.project
        if project is None:
            return set()
        languages: set[str] = set()
        try:
            languages = set(self.rt.index.languages(project.id))
            if not languages:
                profile = self.rt.projects.get(project.id).profile or {}
                languages = set((profile.get("languages") or {}).keys())
        except (CoreError, sqlite3.Error, RuntimeError):
            log.debug("could not read indexed languages", exc_info=True)
        if not languages:
            languages = set(self._scan_languages(Path(project.root_path)))
        return languages

    def _scan_languages(self, root: Path) -> frozenset[str]:
        cached = self._scanned.get(str(root))
        if cached is not None:
            return cached
        found: set[str] = set()
        seen = 0
        for _dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
            for name in filenames:
                language = detect_language(name)
                if language is not None:
                    found.add(language)
            seen += len(filenames)
            if seen >= LANGUAGE_SCAN_LIMIT:
                break
        result = frozenset(found)
        self._scanned[str(root)] = result
        return result

    # ------------------------------------------------------------------ extension API
    def tools_for_role(self, role: str) -> list[Tool]:
        if not self.enabled or self._closed or role not in ROLES:
            return []
        languages = self._project_languages()
        if not any(cand.available and languages.intersection(cand.serves) for cand in self._candidates):
            return []
        return [tool(self) for tool in LSP_TOOLS]

    def status(self) -> dict[str, Any]:
        """Configured and built-in servers, availability reasons and live clients (for doctor)."""
        servers = []
        for cand in self._candidates:
            clients = [c.describe() for (name, _), c in self._clients.items() if name == cand.spec.name]
            servers.append(
                {
                    "name": cand.spec.name,
                    "source": cand.spec.source,
                    "command": list(cand.spec.command),
                    "languages": list(cand.spec.languages),
                    "serves": list(cand.serves),
                    "executable": cand.executable,
                    "available": self.enabled and cand.available,
                    "reason": cand.reason if self.enabled else "LSP disabled by configuration",
                    "clients": clients,
                }
            )
        served = {lang for cand in self._candidates for lang in cand.serves}
        languages: dict[str, dict[str, Any]] = {}
        for lang in sorted(self._project_languages() & (CODE_LANGUAGES | served)):
            chosen = next((c for c in self._candidates if c.available and lang in c.serves), None)
            if not self.enabled:
                languages[lang] = {"server": None, "reason": "LSP disabled by configuration"}
            elif chosen is not None:
                languages[lang] = {"server": chosen.spec.name}
            else:
                reasons = [f"{c.spec.name}: {c.reason}" for c in self._candidates if lang in c.serves]
                languages[lang] = {
                    "server": None,
                    "reason": "; ".join(reasons)
                    if reasons
                    else "no configured or built-in server handles it",
                }
        out: dict[str, Any] = {
            "enabled": self.enabled,
            "servers": servers,
            "project_languages": languages,
            "tools_offered": self.enabled
            and not self._closed
            and any(v.get("server") for v in languages.values()),
        }
        if not self.enabled:
            out["reason"] = "disabled by configuration (intel.lsp_enabled = false)"
        return out

    async def aclose(self) -> None:
        self._closed = True
        clients = list(self._clients.values())
        self._clients.clear()
        results = await asyncio.gather(
            *(client.aclose() for client in clients), *list(self._background), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                log.warning("language server shutdown failed: %s", result)

    # ------------------------------------------------------------------ runtime hooks
    def _process_started(self, conn: RpcConnection) -> str | None:
        pid = conn.pid
        if pid is None:
            return None
        row_id = new_id("prc", now=self.rt.clock.now())
        try:
            self.rt.db.execute(
                "INSERT INTO processes(id, runtime_id, task_id, attempt_id, tool_call_id, pid, pgid, start_ticks, "
                "command, cwd, status, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row_id,
                    self.rt.runtime_id,
                    None,
                    None,
                    None,
                    pid,
                    pid,
                    _start_ticks(pid),  # same format recovery compares against
                    shlex.join(conn.argv)[:4000],
                    str(conn.cwd),
                    "running",
                    self.rt.clock.now(),
                ),
            )
        except (sqlite3.Error, RuntimeError):
            log.warning("could not record language server process %s", pid, exc_info=True)
            return None
        return row_id

    def _process_ended(self, conn: RpcConnection) -> None:
        if conn.row_id is None or conn.proc is None:
            return
        code = conn.proc.returncode
        status = "exited" if code is not None and code >= 0 else "killed"
        try:
            self.rt.db.execute(
                "UPDATE processes SET status = ?, exit_code = ?, ended_at = ? WHERE id = ?",
                (status, code, self.rt.clock.now(), conn.row_id),
            )
        except (sqlite3.Error, RuntimeError):
            log.warning("could not record language server exit for %s", conn.pid, exc_info=True)

    def _event(self, kind: str, level: str, data: dict[str, Any]) -> None:
        project_id = self.rt.project.id if self.rt.project is not None else None
        try:
            if level == "debug":
                self.rt.events.ephemeral(kind, project_id=project_id, data=data)
            else:
                self.rt.events.emit(kind, project_id=project_id, level=level, data=data)
        except (sqlite3.Error, RuntimeError):
            log.warning("could not record %s event", kind, exc_info=True)


# ------------------------------------------------------------------------------- tools
_POSITION_HELP = (
    "Position: `line` and `character` are both 1-based (line numbers as in read_file, columns counted in "
    "characters as in path:line:col results), or pass `symbol` to locate an identifier by name."
)


class _PositionInput(ToolInput):
    path: str = Field(description="File path relative to the workspace root")
    line: int | None = Field(None, ge=1, description="1-based line number")
    character: int | None = Field(
        None,
        ge=1,
        description="1-based column of the identifier on that line (optional when symbol is given)",
    )
    symbol: str | None = Field(
        None,
        min_length=1,
        max_length=200,
        description=(
            "Identifier to position on instead of an exact column. With `line`: its occurrence on that line "
            "(nearest to `character` if given). Without `line`: its definition in the file, else its first occurrence."
        ),
    )

    @model_validator(mode="after")
    def _position_or_symbol(self) -> _PositionInput:
        if self.symbol is None and (self.line is None or self.character is None):
            raise ValueError("give line and character (both 1-based), or a symbol name")
        return self


class _LSPTool(Tool):
    capability = Capability.FS_READ
    timeout_s = 180.0

    def __init__(self, manager: LSPManager):
        self.manager = manager

    def target(self, args: Any) -> str:
        return str(args.path)

    def effective_timeout(self, args: Any) -> float:
        s = self.manager.settings
        wait = float(getattr(args, "wait_seconds", 0.0) or 0.0)
        return s.startup_timeout_s + 2 * s.request_timeout_s + wait + 10.0


class LSPDefinition(_LSPTool):
    name = "lsp_definition"
    description = (
        "Go to the definition of the symbol at a position using the project's language server (compiler-accurate: "
        f"follows imports, aliases and re-exports). {_POSITION_HELP} Returns path:line:col locations with a preview."
    )
    Input = _PositionInput

    async def run(self, ctx: ToolContext, args: _PositionInput) -> ToolResult:
        async with self.manager.document(ctx, args.path) as (client, doc):
            line, column, note = resolve_position(doc.lines, doc.rel, args.line, args.character, args.symbol)
            locations = await client.definition(doc, line, column)
            shown = await self.manager.describe_locations(ctx, client, locations[:MAX_DEFINITIONS])
        subject = _subject(doc, line, column)
        data: dict[str, Any] = {"server": client.name, "locations": shown, "resolved": note}
        if not locations:
            return ToolResult(True, f"{client.name}: no definition found for {subject}", data=data)
        lines = [f"definition of {subject} via {client.name}: {len(locations)} location(s)"]
        if note:
            lines[0] += f" [{note}]"
        lines += [format_location(d) for d in shown]
        if len(locations) > len(shown):
            lines.append(f"… {len(locations) - len(shown)} more not shown")
        return ToolResult(True, "\n".join(lines), data=data)


class LSPReferences(_LSPTool):
    name = "lsp_references"
    description = (
        "Find every reference to the symbol at a position using the language server (semantic, unlike text search: "
        f"no false matches from comments or unrelated names). {_POSITION_HELP}"
    )

    class Input(_PositionInput):
        include_declaration: bool = Field(True, description="Include the declaration itself")
        max_results: int = Field(100, ge=1, le=500, description="Maximum locations to show")

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        async with self.manager.document(ctx, args.path) as (client, doc):
            line, column, note = resolve_position(doc.lines, doc.rel, args.line, args.character, args.symbol)
            locations = await client.references(
                doc, line, column, include_declaration=args.include_declaration
            )
            locations.sort(key=lambda loc: (loc.uri, loc.line, loc.character))
            shown = await self.manager.describe_locations(ctx, client, locations[: args.max_results])
        subject = _subject(doc, line, column)
        files = len({loc.uri for loc in locations})
        data: dict[str, Any] = {
            "server": client.name,
            "count": len(locations),
            "files": files,
            "locations": shown,
            "resolved": note,
        }
        if not locations:
            return ToolResult(True, f"{client.name}: no references found for {subject}", data=data)
        lines = [f"references to {subject} via {client.name}: {len(locations)} in {files} file(s)"]
        if note:
            lines[0] += f" [{note}]"
        lines += [format_location(d) for d in shown]
        if len(locations) > len(shown):
            lines.append(f"… {len(locations) - len(shown)} more not shown (raise max_results, up to 500)")
        return ToolResult(True, "\n".join(lines), data=data)


class LSPHover(_LSPTool):
    name = "lsp_hover"
    description = (
        "Show the type signature and documentation of the symbol at a position (language server hover). "
        + _POSITION_HELP
    )
    Input = _PositionInput

    async def run(self, ctx: ToolContext, args: _PositionInput) -> ToolResult:
        async with self.manager.document(ctx, args.path) as (client, doc):
            line, column, note = resolve_position(doc.lines, doc.rel, args.line, args.character, args.symbol)
            text = (await client.hover(doc, line, column)).strip()
        subject = _subject(doc, line, column)
        data: dict[str, Any] = {"server": client.name, "resolved": note, "found": bool(text)}
        if not text:
            return ToolResult(True, f"{client.name}: no hover information for {subject}", data=data)
        header = f"hover for {subject} via {client.name}" + (f" [{note}]" if note else "")
        return ToolResult(True, f"{header}:\n{truncate_middle(text, MAX_HOVER_CHARS)}", data=data)


class LSPDocumentSymbols(_LSPTool):
    name = "lsp_document_symbols"
    description = (
        "Outline of one file from the language server: classes, functions, methods, fields with 1-based line:col, "
        "nested by containment."
    )

    class Input(ToolInput):
        path: str = Field(description="File path relative to the workspace root")

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        async with self.manager.document(ctx, args.path) as (client, doc):
            symbols = await client.document_symbols(doc)
            entries = flatten_symbols(symbols, doc.lines, client.encoding)
        data: dict[str, Any] = {
            "server": client.name,
            "count": len(entries),
            "symbols": entries[:MAX_SYMBOLS],
        }
        if not entries:
            return ToolResult(True, f"{client.name}: no symbols reported for {doc.rel}", data=data)
        lines = [f"{doc.rel}: {len(entries)} symbol(s) via {client.name}"]
        for e in entries[:MAX_SYMBOLS]:
            text = f"{'  ' * e['depth']}{e['line']}:{e['column']} {e['kind']} {e['name']}"
            if e["detail"]:
                text += f" — {e['detail']}"
            if e["container"]:
                text += f" (in {e['container']})"
            lines.append(text)
        if len(entries) > MAX_SYMBOLS:
            lines.append(f"… {len(entries) - MAX_SYMBOLS} more not shown")
        return ToolResult(True, "\n".join(lines), data=data)


class LSPDiagnostics(_LSPTool):
    name = "lsp_diagnostics"
    description = (
        "Compiler/type-checker diagnostics (errors, warnings) for one file from the language server, after syncing "
        "the file's current content from disk. Reports honestly when the server published nothing in time."
    )

    class Input(ToolInput):
        path: str = Field(description="File path relative to the workspace root")
        wait_seconds: float = Field(
            5.0, ge=0, le=60, description="How long to wait for the server to publish diagnostics"
        )

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        async with self.manager.document(ctx, args.path) as (client, doc):
            result = await client.diagnostics(doc, wait_s=args.wait_seconds)
            entries = [_diagnostic_entry(d, doc.lines, client.encoding) for d in result.items]
        entries.sort(key=lambda e: (e["line"], e["column"], e["severity"]))
        counts: dict[str, int] = {}
        for e in entries:
            counts[e["severity"]] = counts.get(e["severity"], 0) + 1
        data: dict[str, Any] = {
            "server": client.name,
            "mode": result.mode,
            "received": result.received,
            "fresh": result.fresh,
            "counts": counts,
            "diagnostics": entries[:MAX_DIAGNOSTICS],
        }
        if not result.received:
            return ToolResult(
                True,
                f"{client.name} published no diagnostics for {doc.rel} within {args.wait_seconds:g}s; it may still "
                "be analyzing. This is not evidence that the file is clean (retry with a larger wait_seconds).",
                data=data,
            )
        staleness = "" if result.fresh else "; published before the latest sync and not refreshed in time"
        if not entries:
            return ToolResult(
                True, f"{doc.rel}: no diagnostics from {client.name} ({result.mode}{staleness})", data=data
            )
        summary = ", ".join(f"{n} {sev}{'s' if n != 1 else ''}" for sev, n in sorted(counts.items()))
        lines = [f"{doc.rel}: {summary} from {client.name} ({result.mode}{staleness})"]
        for e in entries[:MAX_DIAGNOSTICS]:
            tag = " ".join(part for part in (e["source"], e["code"]) if part)
            lines.append(
                f"{doc.rel}:{e['line']}:{e['column']} {e['severity']}"
                + (f" [{tag}]" if tag else "")
                + f" {e['message']}"
            )
        if len(entries) > MAX_DIAGNOSTICS:
            lines.append(f"… {len(entries) - MAX_DIAGNOSTICS} more not shown")
        return ToolResult(True, "\n".join(lines), data=data)


LSP_TOOLS: tuple[type[_LSPTool], ...] = (
    LSPDefinition,
    LSPReferences,
    LSPHover,
    LSPDiagnostics,
    LSPDocumentSymbols,
)
