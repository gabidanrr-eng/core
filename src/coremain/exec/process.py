"""Subprocess execution with process groups, timeouts, cancellation and bounded capture.

* Every command runs in its own session/process group so timeouts and cancellation terminate
  the whole tree (SIGTERM, grace period, SIGKILL).
* Stragglers left in the group after the main process exits are terminated too.
* stdout/stderr are captured concurrently with a head+tail bound; the full combined output
  is spooled to a temp file when it exceeds the bound so it can become an artifact.
* Every spawned process is registered in the ``processes`` table (pid, pgid, start time
  ticks) so a later runtime can detect and safely terminate orphans after a crash.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import shlex
import shutil
import signal
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from coremain.errors import OperationCancelled, ToolError
from coremain.runtime.cancel import CancelToken
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id

OutputCallback = Callable[[str, str], None]


@dataclass
class ProcessResult:
    command: str
    cwd: str
    status: str  # exited | timeout | cancelled
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    truncated: bool = False
    spool_path: Path | None = None
    process_id: str | None = None
    signal: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "exited" and self.exit_code == 0

    def combined(self, limit: int = 20000) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(("[stderr]\n" if self.stdout else "") + self.stderr)
        text = "\n".join(parts)
        if len(text) > limit:
            head = limit // 3
            text = text[:head] + f"\n… [{len(text) - limit} chars omitted] …\n" + text[-(limit - head):]
        return text


class _Capture:
    def __init__(self, limit: int):
        self.limit = max(2048, limit)
        self.head: list[str] = []
        self.head_len = 0
        self.tail: list[str] = []
        self.tail_len = 0
        self.total_bytes = 0
        self.truncated = False

    def add(self, text: str, nbytes: int) -> None:
        self.total_bytes += nbytes
        half = self.limit // 2
        if self.head_len < half:
            take = text[: half - self.head_len]
            self.head.append(take)
            self.head_len += len(take)
            text = text[len(take):]
        if not text:
            return
        self.tail.append(text)
        self.tail_len += len(text)
        while self.tail_len > half and len(self.tail) > 1:
            dropped = self.tail.pop(0)
            self.tail_len -= len(dropped)
            self.truncated = True
        if self.tail_len > half:
            joined = "".join(self.tail)
            self.tail = [joined[-half:]]
            self.tail_len = half
            self.truncated = True

    def text(self) -> str:
        body = "".join(self.head)
        if self.truncated:
            body += f"\n… [output truncated; {self.total_bytes} bytes total] …\n"
        return body + "".join(self.tail)


def _start_ticks(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def process_alive(pid: int, start_ticks: str | None = None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if start_ticks is not None:
        current = _start_ticks(pid)
        if current is not None and current != start_ticks:
            return False  # PID was reused by an unrelated process
    return True


def kill_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class ProcessRunner:
    def __init__(
        self,
        db: Database | None,
        clock: Clock,
        runtime_id: str,
        *,
        max_output_bytes: int = 4_000_000,
        kill_grace_s: float = 3.0,
        max_memory_mb: int | None = None,
        spool_dir: Path | None = None,
    ):
        self.db = db
        self.clock = clock
        self.runtime_id = runtime_id
        self.max_output_bytes = max_output_bytes
        self.kill_grace_s = kill_grace_s
        self.max_memory_mb = max_memory_mb
        self.spool_dir = spool_dir
        self._active: dict[str, asyncio.subprocess.Process] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    def _register(self, proc_id: str, pid: int, command: str, cwd: Path, task_id: str | None, attempt_id: str | None,
                  tool_call_id: str | None) -> None:
        if self.db is None:
            return
        self.db.execute(
            "INSERT INTO processes(id, runtime_id, task_id, attempt_id, tool_call_id, pid, pgid, start_ticks, command, cwd, status, started_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (proc_id, self.runtime_id, task_id, attempt_id, tool_call_id, pid, pid, _start_ticks(pid), command[:4000], str(cwd),
             "running", self.clock.now()),
        )

    def _finish(self, proc_id: str, status: str, exit_code: int | None) -> None:
        if self.db is None:
            return
        self.db.execute("UPDATE processes SET status = ?, exit_code = ?, ended_at = ? WHERE id = ?",
                        (status, exit_code, self.clock.now(), proc_id))

    async def run(
        self,
        argv: list[str] | None = None,
        *,
        shell_command: str | None = None,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: float,
        cancel: CancelToken | None = None,
        on_output: OutputCallback | None = None,
        stdin_data: bytes | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        tool_call_id: str | None = None,
        kill_stragglers: bool = True,
    ) -> ProcessResult:
        if (argv is None) == (shell_command is None):
            raise ValueError("exactly one of argv or shell_command is required")
        if cancel is not None:
            cancel.raise_if_cancelled()
        if shell_command is not None:
            shell = shutil.which("bash") or "/bin/sh"
            exec_argv = [shell, "-c", shell_command]
            display = shell_command
        else:
            assert argv is not None
            exec_argv = list(argv)
            display = shlex.join(argv)
        if self.max_memory_mb and shutil.which("prlimit"):
            exec_argv = ["prlimit", f"--as={self.max_memory_mb * 1024 * 1024}", "--", *exec_argv]
        if not os.path.isdir(cwd):
            raise ToolError(f"working directory does not exist: {cwd}", error_class="invalid_cwd")
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *exec_argv,
                cwd=str(cwd),
                env=dict(env),
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"command not found: {exec_argv[0]}", error_class="command_not_found") from exc
        except PermissionError as exc:
            raise ToolError(f"permission denied executing {exec_argv[0]}", error_class="permission_denied") from exc
        proc_id = new_id("prc")
        self._register(proc_id, proc.pid, display, cwd, task_id, attempt_id, tool_call_id)
        self._active[proc_id] = proc
        pgid = proc.pid
        out_cap = _Capture(self.max_output_bytes // 2)
        err_cap = _Capture(self.max_output_bytes // 2)
        spool_fh = None
        spool_path: Path | None = None
        if self.spool_dir is not None:
            self.spool_dir.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix="proc-", suffix=".log", dir=self.spool_dir)
            spool_fh = os.fdopen(fd, "w", encoding="utf-8", errors="replace")
            spool_path = Path(name)

        async def pump(stream: asyncio.StreamReader | None, name: str, cap: _Capture) -> None:
            if stream is None:
                return
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                chunk = await stream.read(8192)
                if not chunk:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        cap.add(tail, 0)
                    return
                text = decoder.decode(chunk)
                cap.add(text, len(chunk))
                if spool_fh is not None:
                    spool_fh.write(text)
                if on_output is not None:
                    with contextlib.suppress(Exception):
                        on_output(name, text)

        if stdin_data is not None and proc.stdin is not None:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                proc.stdin.write(stdin_data)
                await proc.stdin.drain()
                proc.stdin.close()

        pumps = asyncio.gather(pump(proc.stdout, "stdout", out_cap), pump(proc.stderr, "stderr", err_cap))
        status = "exited"
        notes: list[str] = []
        waiter = asyncio.ensure_future(proc.wait())
        cancel_waiter = asyncio.ensure_future(cancel.wait()) if cancel is not None else None
        try:
            waits = {waiter} | ({cancel_waiter} if cancel_waiter else set())
            done, _ = await asyncio.wait(waits, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
            if waiter not in done:
                status = "cancelled" if cancel_waiter is not None and cancel_waiter in done else "timeout"
                await self._terminate(proc, pgid)
                notes.append(f"process group terminated ({status})")
        except asyncio.CancelledError:
            await self._terminate(proc, pgid)
            self._finish(proc_id, "killed", None)
            self._active.pop(proc_id, None)
            pumps.cancel()
            raise
        finally:
            if cancel_waiter is not None:
                cancel_waiter.cancel()
        exit_code = await waiter
        if kill_stragglers and kill_group(pgid, 0):
            # Children still alive in the group after the main process exited.
            kill_group(pgid, signal.SIGTERM)
            await asyncio.sleep(0.2)
            if kill_group(pgid, 0):
                kill_group(pgid, signal.SIGKILL)
            notes.append("terminated leftover background processes")
        try:
            await asyncio.wait_for(pumps, timeout=5)
        except TimeoutError:
            notes.append("output streams did not close; capture may be incomplete")
            pumps.cancel()
        if spool_fh is not None:
            spool_fh.close()
        truncated = out_cap.truncated or err_cap.truncated
        if spool_path is not None and not truncated:
            spool_path.unlink(missing_ok=True)
            spool_path = None
        final_status = {"exited": "exited", "timeout": "timeout", "cancelled": "killed"}[status]
        self._finish(proc_id, final_status, exit_code)
        self._active.pop(proc_id, None)
        sig = -exit_code if exit_code is not None and exit_code < 0 else None
        result = ProcessResult(
            command=display, cwd=str(cwd), status=status, exit_code=exit_code, stdout=out_cap.text(), stderr=err_cap.text(),
            duration_s=time.monotonic() - started, stdout_bytes=out_cap.total_bytes, stderr_bytes=err_cap.total_bytes,
            truncated=truncated, spool_path=spool_path, process_id=proc_id, signal=sig, notes=notes,
        )
        if status == "cancelled" and cancel is not None:
            raise OperationCancelled(cancel.reason or "cancelled", details={"command": display})
        return result

    async def _terminate(self, proc: asyncio.subprocess.Process, pgid: int) -> None:
        if proc.returncode is not None and not kill_group(pgid, 0):
            return
        kill_group(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.kill_grace_s)
        except TimeoutError:
            pass
        if kill_group(pgid, 0):
            kill_group(pgid, signal.SIGKILL)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=self.kill_grace_s)

    async def terminate_all(self) -> int:
        count = 0
        for proc_id, proc in list(self._active.items()):
            await self._terminate(proc, proc.pid)
            self._finish(proc_id, "killed", proc.returncode)
            self._active.pop(proc_id, None)
            count += 1
        return count
