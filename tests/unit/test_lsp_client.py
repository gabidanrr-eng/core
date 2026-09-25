"""LSP protocol and client behaviour against a real (fake) language server process.

Covers framing, position encodings, request timeouts and cancellation, server->client requests,
crash detection with bounded restarts, startup failures and process-group cleanup.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.fixtures.lsp_support import (
    FAKE_SERVER,
    PROJECT_FILES,
    events,
    gone,
    read_log,
    received,
    wait_for_log,
    wait_gone,
)
from tests.helpers import write_files

from coremain.errors import ToolError
from coremain.intel.lsp import (
    LSP_CRASHED,
    LSP_PROTOCOL,
    LSP_TIMEOUT,
    LSP_UNAVAILABLE,
    STDERR_LINES,
    LanguageServerClient,
    LSPError,
    LSPSettings,
    ServerSpec,
    encode_message,
    from_lsp_character,
    hover_text,
    lsp_root,
    parse_locations,
    read_message,
    resolve_position,
    to_lsp_character,
)

FAST = {"request_timeout_s": 5.0, "startup_timeout_s": 10.0, "exit_wait_s": 1.0, "kill_grace_s": 1.0}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    write_files(root, PROJECT_FILES)
    return root


_CREATED: list[LanguageServerClient] = []


def make_client(project: Path, log: Path, *flags: str, **settings: float) -> LanguageServerClient:
    argv = [sys.executable, str(FAKE_SERVER), "--log", str(log), *flags]
    spec = ServerSpec("fake", tuple(argv), ("python",))
    client = LanguageServerClient(
        spec, project, argv=argv, env=dict(os.environ), settings=LSPSettings(**{**FAST, **settings})
    )
    _CREATED.append(client)
    return client


@pytest.fixture(autouse=True)
async def close_clients() -> AsyncIterator[None]:
    """Close every client a test created, so a failing assertion never leaks a server process."""
    yield
    while _CREATED:
        await _CREATED.pop().aclose()


def feed(*chunks: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    for chunk in chunks:
        reader.feed_data(chunk)
    reader.feed_eof()
    return reader


# ------------------------------------------------------------------------------ framing
async def test_framing_round_trip_multiple_messages_and_clean_eof() -> None:
    first = {"jsonrpc": "2.0", "id": 1, "result": {"text": "é🙂 ünïcode"}}
    second = {"jsonrpc": "2.0", "method": "x"}
    reader = feed(encode_message(first) + encode_message(second))
    assert await read_message(reader) == first
    assert await read_message(reader) == second
    assert await read_message(reader) is None


async def test_framing_tolerates_case_extra_headers_and_stray_output() -> None:
    body = b'{"jsonrpc":"2.0","id":7,"result":null}'
    noise: list[str] = []
    raw = (
        b"server starting up\r\n"
        + b"content-length: %d\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n" % len(body)
        + body
    )
    assert await read_message(feed(raw), noise=noise.append) == {"jsonrpc": "2.0", "id": 7, "result": None}
    assert noise == ["server starting up"]


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        (b"Content-Type: x\r\n\r\n{}", "without Content-Length"),
        (b"Content-Length: abc\r\n\r\n{}", "invalid Content-Length"),
        (b"Content-Length: 10\r\n\r\n{}", "middle of a message"),
        (b"Content-Length: 3\r\n\r\n{x}", "invalid JSON"),
        (b"Content-Length: 99999999999\r\n\r\n", "exceeds"),
        (b"Content-Length: 2\r\n", "middle of a message"),
    ],
)
async def test_framing_errors_are_protocol_errors(raw: bytes, fragment: str) -> None:
    with pytest.raises(LSPError) as info:
        await read_message(feed(raw))
    assert info.value.error_class == LSP_PROTOCOL
    assert fragment in info.value.message


# ---------------------------------------------------------------------------- positions
def test_position_encodings_round_trip_through_non_bmp_text() -> None:
    line = 's = "é🙂"; print(add(1, 2))'
    col = line.index("add")
    assert col == 16
    assert to_lsp_character(line, col, "utf-16") == 17  # 🙂 is a surrogate pair
    assert to_lsp_character(line, col, "utf-8") == 20  # é is 2 bytes, 🙂 is 4
    assert to_lsp_character(line, col, "utf-32") == 16
    for encoding in ("utf-16", "utf-8", "utf-32"):
        assert from_lsp_character(line, to_lsp_character(line, col, encoding), encoding) == col
    assert from_lsp_character(line, 10_000, "utf-16") == len(line)
    assert from_lsp_character("", 3, "utf-8") == 0


def test_resolve_position_explicit_symbol_and_errors() -> None:
    lines = PROJECT_FILES["main.py"].split("\n")
    assert resolve_position(lines, "main.py", 3, 17, None) == (2, 16, None)
    assert resolve_position(lines, "main.py", 3, None, "add")[:2] == (2, 16)
    assert resolve_position(lines, "main.py", 1, 40, "add")[:2] == (0, 29)  # nearest occurrence on the line
    line, column, note = resolve_position(lines, "main.py", None, None, "calc.add")
    assert (line, column) == (0, 29) and note is not None and "first occurrence" in note
    calc = PROJECT_FILES["calc.py"].split("\n")
    line, column, note = resolve_position(calc, "calc.py", None, None, "total")
    assert (line, column) == (6, 8) and note is not None and "definition" in note
    for args, klass in [
        ((99, 1, None), "invalid_position"),
        ((1, 500, None), "invalid_position"),
        ((1, None, "nothere"), "symbol_not_found"),
        ((None, None, "nothere"), "symbol_not_found"),
        ((None, None, "..."), "invalid_arguments"),
    ]:
        with pytest.raises(ToolError) as info:
            resolve_position(lines, "main.py", *args)
        assert info.value.error_class == klass


def test_lsp_root_uses_outermost_marker_inside_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    write_files(ws, {"svc/api/pyproject.toml": "", "svc/api/pkg/mod.py": "", "tool.py": ""})
    assert lsp_root(ws, ws / "svc/api/pkg/mod.py", ("pyproject.toml",)) == (ws / "svc/api").resolve()
    write_files(ws, {"pyproject.toml": ""})
    assert lsp_root(ws, ws / "svc/api/pkg/mod.py", ("pyproject.toml",)) == ws.resolve()
    assert lsp_root(ws, ws / "tool.py", ()) == ws.resolve()
    write_files(tmp_path, {"pyproject.toml": ""})  # markers above the workspace are ignored
    assert lsp_root(ws / "svc", ws / "svc/other.py", ("pyproject.toml",)) == (ws / "svc").resolve()


def test_parse_locations_and_hover_shapes() -> None:
    rng = {"start": {"line": 1, "character": 2}, "end": {"line": 1, "character": 5}}
    single = parse_locations({"uri": "file:///a%20b.py", "range": rng})
    assert [(loc.uri, loc.line, loc.character) for loc in single] == [("file:///a%20b.py", 1, 2)]
    links = parse_locations(
        [{"targetUri": "file:///x.py", "targetRange": rng, "targetSelectionRange": rng}] * 2
    )
    assert len(links) == 1  # de-duplicated
    assert parse_locations(None) == [] and parse_locations([{"uri": 3}]) == []
    assert hover_text({"kind": "markdown", "value": "**x**"}) == "**x**"
    assert hover_text([{"language": "python", "value": "def f()"}, "doc"]) == "```python\ndef f()\n```\n\ndoc"
    assert hover_text(None) == ""


# ------------------------------------------------------------------------ live protocol
async def test_queries_sync_versions_and_graceful_shutdown(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log)
    doc = await client.sync(project / "main.py", "main.py", "python")
    assert client.state == "ready" and client.encoding == "utf-16"
    assert client.server_info == {"name": "fake-lsp", "version": "1.0"}
    definitions = await client.definition(doc, 2, 16)
    assert [(Path(d.uri).name, d.line, d.character) for d in definitions] == [("calc.py", 0, 4)]
    assert "def add(a, b)" in await client.hover(doc, 2, 16)
    # Unchanged content is not re-sent; changed content goes out as didChange with the next version.
    assert await client.sync(project / "main.py", "main.py", "python") is doc
    (project / "main.py").write_text("value = 1\n", encoding="utf-8")
    doc = await client.sync(project / "main.py", "main.py", "python")
    assert doc.version == 2
    # didChange is a notification: nothing orders it before this read except waiting for it.
    await wait_for_log(
        log, lambda es: any(e.get("in", {}).get("method") == "textDocument/didChange" for e in es)
    )
    opens = received(log, "textDocument/didOpen")
    changes = received(log, "textDocument/didChange")
    assert len(opens) == 1 and opens[0]["params"]["textDocument"]["version"] == 1
    assert [c["params"]["textDocument"]["version"] for c in changes] == [2]
    assert changes[0]["params"]["contentChanges"] == [{"text": "value = 1\n"}]
    pid = client.pid
    assert pid is not None
    await client.aclose()
    assert await wait_gone(pid)
    methods = [e["in"].get("method") for e in read_log(log) if "in" in e]
    assert methods[-2:] == ["shutdown", "exit"]
    assert events(log, "exit") == [{"pid": pid, "event": "exit", "code": 0}]
    with pytest.raises(LSPError) as info:
        await client.sync(project / "main.py", "main.py", "python")
    assert info.value.error_class == LSP_UNAVAILABLE


async def test_server_requests_get_conservative_answers(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, "--server-requests")
    try:
        await client.sync(project / "calc.py", "calc.py", "python")

        def answers(entries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
            sent = {e["id"]: e["method"] for e in entries if e.get("event") == "server-request"}
            replies = [e["in"] for e in entries if isinstance(e.get("in"), dict) and "method" not in e["in"]]
            return {str(sent[r["id"]]): r for r in replies if r.get("id") in sent}

        entries = await wait_for_log(log, lambda es: len(answers(es)) == 6)
        by_method = answers(entries)
        assert by_method["workspace/configuration"]["result"] == [None, None]
        assert by_method["client/registerCapability"]["result"] is None
        assert by_method["window/workDoneProgress/create"]["result"] is None
        assert by_method["workspace/workspaceFolders"]["result"] == [
            {"uri": project.resolve().as_uri(), "name": "proj"}
        ]
        edit = by_method["workspace/applyEdit"]["result"]
        assert isinstance(edit, dict) and edit["applied"] is False
        error = by_method["custom/unknownRequest"]["error"]
        assert isinstance(error, dict) and error["code"] == -32601
        assert "HACKED" not in (project / "calc.py").read_text()
        assert client.messages and "example error log" in client.messages[-1]
        assert client.describe()["indexing"] is False  # progress begin/end tracked
    finally:
        await client.aclose()


async def test_timeout_sends_cancel_request_and_connection_survives(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, "--hang-on", "textDocument/hover", request_timeout_s=0.4)
    try:
        doc = await client.sync(project / "main.py", "main.py", "python")
        started = time.monotonic()
        with pytest.raises(LSPError) as info:
            await client.hover(doc, 2, 16)
        assert info.value.error_class == LSP_TIMEOUT
        assert time.monotonic() - started < 3
        hover_id = received(log, "textDocument/hover")[0]["id"]
        await wait_for_log(
            log, lambda es: any(e.get("event") == "cancelled" and e["id"] == hover_id for e in es)
        )
        assert received(log, "$/cancelRequest")[0]["params"] == {"id": hover_id}
        # The late RequestCancelled answer is ignored and the same process keeps serving.
        pid = client.pid
        assert await client.definition(doc, 2, 16)
        assert client.pid == pid and client.conn is not None and client.conn.pending_count == 0
    finally:
        await client.aclose()


async def test_cancelling_an_awaited_request_notifies_the_server(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, "--hang-on", "textDocument/hover")
    try:
        doc = await client.sync(project / "main.py", "main.py", "python")
        task = asyncio.create_task(client.hover(doc, 2, 16))
        await wait_for_log(
            log, lambda es: any(e.get("in", {}).get("method") == "textDocument/hover" for e in es)
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_for_log(log, lambda es: any(e.get("event") == "cancelled" for e in es))
        assert client.conn is not None and client.conn.pending_count == 0
    finally:
        await client.aclose()


async def test_crash_fails_pending_request_then_restarts_with_backoff(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(
        project,
        log,
        "--crash-on",
        "textDocument/definition",
        "--crash-times",
        "1",
        "--state-file",
        str(tmp_path / "crashes"),
        restart_backoff_s=0.4,
    )
    try:
        doc = await client.sync(project / "main.py", "main.py", "python")
        first_pid = client.pid
        assert first_pid is not None
        with pytest.raises(LSPError) as info:
            await client.definition(doc, 2, 16)
        assert info.value.error_class == LSP_CRASHED
        assert "exited with code 3" in info.value.message
        assert "fake-lsp: crashing on textDocument/definition" in info.value.message
        assert client.state == "crashed" and client.crashes == 1
        assert await wait_gone(first_pid)
        started = time.monotonic()
        doc = await client.sync(project / "main.py", "main.py", "python")
        assert time.monotonic() - started >= 0.3  # exponential backoff before the restart
        assert client.pid not in (None, first_pid)
        assert client.restarts == 1 and client.starts == 2
        assert [(Path(d.uri).name, d.line) for d in await client.definition(doc, 2, 16)] == [("calc.py", 0)]
        assert len(received(log, "textDocument/didOpen")) == 2  # documents re-opened on the new process
    finally:
        await client.aclose()


async def test_restart_budget_is_bounded(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(
        project, log, "--crash-on", "textDocument/hover", max_restarts=1, restart_backoff_s=0.01
    )
    for _ in range(2):
        doc = await client.sync(project / "main.py", "main.py", "python")
        with pytest.raises(LSPError) as info:
            await client.hover(doc, 2, 16)
        assert info.value.error_class == LSP_CRASHED
    for _ in range(2):
        with pytest.raises(LSPError) as info:
            await client.sync(project / "main.py", "main.py", "python")
        assert info.value.error_class == LSP_CRASHED
        assert "restart budget (1) is exhausted" in info.value.message
    assert client.state == "failed"
    assert len(received(log, "initialize")) == 2  # no further spawns once the budget is spent
    await client.aclose()


async def test_protocol_violation_kills_server_and_is_classified(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, "--garbage-on", "textDocument/hover", restart_backoff_s=0.01)
    try:
        doc = await client.sync(project / "main.py", "main.py", "python")
        pid = client.pid
        assert pid is not None
        with pytest.raises(LSPError) as info:
            await client.hover(doc, 2, 16)
        assert info.value.error_class == LSP_PROTOCOL
        assert await wait_gone(pid)
        doc = await client.sync(project / "main.py", "main.py", "python")
        assert await client.definition(doc, 2, 16)
    finally:
        await client.aclose()


async def test_startup_timeout_counts_as_crash_and_kills_the_process(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, "--delay-initialize", "30", startup_timeout_s=0.5)
    with pytest.raises(LSPError) as info:
        await client.sync(project / "main.py", "main.py", "python")
    assert info.value.error_class == LSP_TIMEOUT
    assert client.state == "crashed" and client.crashes == 1
    assert "did not initialize within 0.5s" in (client.last_error or "")
    pid = next(e["pid"] for e in read_log(log) if e.get("in", {}).get("method") == "initialize")
    assert await wait_gone(pid)
    await client.aclose()


async def test_missing_executable_fails_permanently_without_retries(project: Path) -> None:
    spec = ServerSpec("ghost", ("/nonexistent/ghost-lsp",), ("python",))
    client = LanguageServerClient(
        spec, project, argv=["/nonexistent/ghost-lsp"], env=dict(os.environ), settings=LSPSettings(**FAST)
    )
    for _ in range(2):
        with pytest.raises(LSPError) as info:
            await client.sync(project / "main.py", "main.py", "python")
        assert info.value.error_class == LSP_UNAVAILABLE
        assert "command not found" in info.value.message
    assert client.state == "failed" and client.starts == 0


async def test_stderr_is_drained_and_bounded(project: Path, tmp_path: Path) -> None:
    client = make_client(project, tmp_path / "lsp.log", "--stderr-noise", "5000")
    try:
        await client.sync(project / "main.py", "main.py", "python")
        conn = client.conn
        assert conn is not None
        deadline = time.monotonic() + 5
        while conn.stderr_bytes < 100_000 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert conn.stderr_bytes > 100_000
        assert "fake-lsp noise line 4999" in conn.stderr_tail(1)
        assert len(conn.stderr_tail(10_000).splitlines()) == STDERR_LINES
    finally:
        await client.aclose()


async def test_server_without_sync_reads_from_disk(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, "--sync-none")
    try:
        doc = await client.sync(project / "main.py", "main.py", "python")
        assert await client.definition(doc, 2, 16)
        assert received(log, "textDocument/didOpen") == []
    finally:
        await client.aclose()


async def test_kill_fallback_reaps_whole_process_group(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, "--ignore-exit", "--spawn-child", exit_wait_s=0.2, kill_grace_s=0.2)
    await client.sync(project / "main.py", "main.py", "python")
    pid = client.pid
    child = events(log, "child")[0]["child_pid"]
    assert pid is not None and not gone(pid) and not gone(child)
    started = time.monotonic()
    await client.aclose()
    assert time.monotonic() - started < 5
    assert await wait_gone(pid) and await wait_gone(child)
    assert events(log, "ignored-exit")  # it really did ignore the graceful path
    assert client.conn is not None and client.conn.proc is not None and client.conn.proc.returncode == -9


async def test_externally_killed_server_counts_one_crash_and_restarts(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, restart_backoff_s=0.01)
    try:
        await client.sync(project / "main.py", "main.py", "python")
        pid = client.pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        assert await wait_gone(pid)
        doc = await client.sync(project / "main.py", "main.py", "python")
        assert client.pid not in (None, pid)
        assert (client.crashes, client.restarts, client.starts) == (1, 1, 2)
        assert "was killed by signal 9" in (client.last_error or "")
        assert await client.definition(doc, 2, 16)
    finally:
        await client.aclose()


async def test_close_during_startup_leaves_no_process_and_no_crash(project: Path, tmp_path: Path) -> None:
    log = tmp_path / "lsp.log"
    client = make_client(project, log, "--delay-initialize", "30")
    task = asyncio.create_task(client.sync(project / "main.py", "main.py", "python"))
    entries = await wait_for_log(
        log, lambda es: any(e.get("in", {}).get("method") == "initialize" for e in es)
    )
    pid = next(e["pid"] for e in entries if e.get("in", {}).get("method") == "initialize")
    await client.aclose()
    with pytest.raises(LSPError) as info:
        await task
    assert info.value.error_class == LSP_UNAVAILABLE
    assert await wait_gone(pid)
    assert client.state == "closed" and client.crashes == 0
