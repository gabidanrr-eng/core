"""`core serve --stdio` as a real subprocess: NDJSON and LSP framing, task lifecycle, events."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from tests.helpers import FIX_CALC_TURNS, Harness, approve_review

SERVE = [sys.executable, "-c", "from coremain.cli.main import main; main()", "serve", "--stdio"]


class Client:
    def __init__(self, proc: asyncio.subprocess.Process, framing: str = "ndjson"):
        self.proc = proc
        self.framing = framing
        self.next_id = 0
        self.notifications: list[dict[str, Any]] = []

    async def _send(self, msg: dict[str, Any]) -> None:
        body = json.dumps(msg).encode()
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            f"Content-Length: {len(body)}\r\n\r\n".encode() + body if self.framing == "lsp" else body + b"\n"
        )
        await self.proc.stdin.drain()

    async def _recv(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        if self.framing == "lsp":
            header = await self.proc.stdout.readuntil(b"\r\n\r\n")
            length = int(header.split(b":")[1].strip().split(b"\r\n")[0])
            return json.loads(await self.proc.stdout.readexactly(length))
        return json.loads(await self.proc.stdout.readline())

    async def call(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 120
    ) -> dict[str, Any]:
        self.next_id += 1
        rid = self.next_id
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            msg = await asyncio.wait_for(self._recv(), timeout=timeout)
            if msg.get("id") == rid:
                return msg
            self.notifications.append(msg)


async def _spawn(harness: Harness, cwd: Path) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *SERVE,
        cwd=cwd,
        env=harness.env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def test_task_lifecycle_over_ndjson(harness: Harness, calc_repo: Path) -> None:
    harness.scripted(
        {
            "name": "fix",
            "roles": {
                "debugger": FIX_CALC_TURNS,
                "implementer": FIX_CALC_TURNS,
                "reviewer": approve_review(),
            },
        }
    )
    proc = await _spawn(harness, calc_repo)
    client = Client(proc)
    try:
        init = (await client.call("initialize"))["result"]
        assert init["framing"] == "ndjson" and init["project"]["root"] == str(calc_repo)
        assert "task.submit" in init["methods"]
        sub = (await client.call("events.subscribe", {"kinds": ["task.", "tool."]}))["result"]["subscription"]
        task = (
            await client.call(
                "task.submit", {"text": "The test_add test is failing; fix the add function in calc.py"}
            )
        )["result"]
        final = (await client.call("task.wait", {"task_id": task["id"], "timeout_s": 90}))["result"]
        assert final["status"] == "completed" and not final["timed_out"]
        evidence = (await client.call("task.evidence", {"task_id": task["id"]}))["result"]
        assert any(e["kind"] == "tests" and e["status"] == "pass" for e in evidence)
        missing = await client.call("task.get", {"task_id": "tsk_does_not_exist"})
        assert missing["error"]["code"] == -32006 and missing["error"]["data"]["error"] == "not_found"
        unknown = await client.call("no.such.method")
        assert unknown["error"]["code"] == -32601
        await client.call("status")  # gives the event pump time to flush
        await asyncio.sleep(0.5)
        await client.call("status")
        events = [
            n
            for n in client.notifications
            if n.get("method") == "event" and n["params"]["subscription"] == sub
        ]
        kinds = {n["params"]["kind"] for n in events}
        assert "task.transition" in kinds and any(k.startswith("tool.") for k in kinds)
        assert (await client.call("shutdown"))["result"] == {"ok": True}
        await asyncio.wait_for(proc.wait(), timeout=30)
        assert proc.returncode == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    assert (calc_repo / "calc.py").read_text().strip().endswith("a + b")


async def test_lsp_framing_and_invalid_requests(harness: Harness, calc_repo: Path) -> None:
    harness.scripted({"name": "none", "roles": {}})
    proc = await _spawn(harness, calc_repo)
    client = Client(proc, framing="lsp")
    try:
        init = (await client.call("initialize"))["result"]
        assert init["framing"] == "lsp" and init["protocol"] == 1
        bad = await client.call("task.submit", {})
        assert bad["error"]["code"] == -32602
        assert proc.stdin is not None
        proc.stdin.close()  # EOF ends the server cleanly
        await asyncio.wait_for(proc.wait(), timeout=30)
        assert proc.returncode == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
