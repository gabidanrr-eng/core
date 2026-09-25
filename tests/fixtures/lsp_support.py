"""Shared helpers for the LSP tests: the fixture project, fake-server commands and process checks."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FAKE_SERVER = Path(__file__).with_name("fake_lsp_server.py")

# main.py line 3 puts a non-BMP character before `add`, so UTF-16/UTF-8 columns differ from
# code-point columns: `add` is at character 17 (1-based) but UTF-16 offset 17 (0-based).
PROJECT_FILES = {
    "calc.py": (
        "def add(a, b):\n"
        '    """Add two numbers."""\n'
        "    return a + b\n"
        "\n"
        "\n"
        "class Calculator:\n"
        "    def total(self, items):\n"
        "        return sum(items)\n"
    ),
    "main.py": (
        'from calc import Calculator, add\n\ns = "é🙂"; print(add(1, 2))\nvalue = undefined_name + 1\n'
    ),
    "pyproject.toml": "[project]\nname = 'demo'\nversion = '0'\n",
}


def fake_command(log: Path, *flags: str) -> list[str]:
    return [sys.executable, str(FAKE_SERVER), "--log", str(log), *flags]


def lsp_toml(
    command: Iterable[str],
    *,
    languages: Iterable[str] = ("python",),
    enabled: bool = True,
    name: str = "fake",
) -> str:
    cmd = ", ".join(json.dumps(part) for part in command)
    langs = ", ".join(json.dumps(lang) for lang in languages)
    return (
        f"[intel]\nlsp_enabled = {'true' if enabled else 'false'}\n\n"
        f"[intel.lsp_servers.{name}]\ncommand = [{cmd}]\nlanguages = [{langs}]\n"
    )


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def received(path: Path, method: str) -> list[dict[str, Any]]:
    return [
        e["in"] for e in read_log(path) if isinstance(e.get("in"), dict) and e["in"].get("method") == method
    ]


def events(path: Path, name: str) -> list[dict[str, Any]]:
    return [e for e in read_log(path) if e.get("event") == name]


async def wait_for_log(path: Path, predicate: Any, timeout: float = 5.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while True:
        entries = read_log(path)
        if predicate(entries):
            return entries
        if time.monotonic() > deadline:
            raise AssertionError(f"log condition not met within {timeout}s; log: {entries[-10:]}")
        await asyncio.sleep(0.05)


def gone(pid: int) -> bool:
    """True when the process no longer runs (absent, or a zombie awaiting its reaper)."""
    if not Path("/proc").is_dir():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state in ("Z", "X")


async def wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not gone(pid):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.05)
    return True
