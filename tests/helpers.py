"""Reusable test helpers: temporary git repositories, isolated Core Main homes, scripted models.

Every test gets its own CORE_HOME and HOME so nothing touches the developer's real state, and the
environment passed to the runtime is an allowlist (provider keys in the outer shell never leak in).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coremain.paths import CorePaths, resolve_core_paths
from coremain.runtime.app import CoreRuntime

PYTEST = f"{sys.executable} -m pytest -q -p no:cacheprovider"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def write_files(root: Path, files: Mapping[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def make_repo(root: Path, files: Mapping[str, str], *, commit: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_files(root, files)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    git(root, "config", "commit.gpgsign", "false")
    if commit:
        git(root, "add", "-A")
        git(root, "commit", "-qm", "init")
    return root


CALC_REPO = {
    "calc.py": "def add(a, b):\n    return a - b\n",
    "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    "pyproject.toml": "[project]\nname = 'proj'\nversion = '0'\n\n[tool.pytest.ini_options]\npythonpath = ['.']\n",
}


def toml_str(value: str) -> str:
    return json.dumps(value)


def turn(*calls: tuple[str, dict[str, Any]], text: str = "") -> dict[str, Any]:
    return {"text": text, "tool_calls": [{"name": name, "arguments": args} for name, args in calls]}


def approve_review(summary: str = "Correct, minimal change") -> list[dict[str, Any]]:
    return [turn(("submit_review", {"verdict": "approve", "summary": summary, "findings": []}))]


FIX_CALC_TURNS = [
    turn(("read_file", {"path": "calc.py"})),
    turn(("edit_file", {"path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"})),
    turn(("run_tests", {})),
    turn(
        (
            "submit_result",
            {
                "summary": "Fixed add() to add instead of subtract",
                "files_changed": ["calc.py"],
                "confidence": "high",
            },
        )
    ),
]


@dataclass
class Harness:
    tmp: Path
    env: dict[str, str] = field(default_factory=dict)
    _scripts: int = 0

    def __post_init__(self) -> None:
        keep = ("PATH", "LANG", "LC_ALL", "TMPDIR", "TERM")
        userhome = self.tmp / "userhome"
        userhome.mkdir(parents=True, exist_ok=True)
        self.env = {k: os.environ[k] for k in keep if k in os.environ}
        self.env.update({"HOME": str(userhome), "CORE_HOME": str(self.tmp / "core")})

    @property
    def home(self) -> Path:
        return self.tmp / "core"

    @property
    def paths(self) -> CorePaths:
        return resolve_core_paths(self.env)

    @property
    def config_file(self) -> Path:
        return self.paths.config_file

    def config(self, text: str) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(text, encoding="utf-8")

    def write_script(self, script: Mapping[str, Any]) -> Path:
        self._scripts += 1
        path = self.tmp / f"script-{self._scripts}.json"
        path.write_text(json.dumps(script), encoding="utf-8")
        return path

    def scripted(
        self,
        script: Mapping[str, Any],
        *,
        models: Iterable[str] = ("scripted",),
        test_command: str | None = PYTEST,
        model_extra: Mapping[str, str] | None = None,
        extra: str = "",
    ) -> Path:
        """Configure one scripted provider serving ``models`` (all replaying the same script)."""
        path = self.write_script(script)
        lines = ["[providers.local]", 'kind = "scripted"', f"script = {toml_str(str(path))}", ""]
        for key in models:
            lines += [
                f"[models.{key}]",
                'provider = "local"',
                f"id = {toml_str(key)}",
                "context_window = 100000",
                "max_output_tokens = 4096",
                *(f"{k} = {v}" for k, v in (model_extra or {}).items()),
                "",
            ]
        if test_command:
            lines += ["[verification.commands]", f"test = {toml_str(test_command)}", ""]
        self.config("\n".join(lines) + "\n" + extra)
        return path

    @contextlib.asynccontextmanager
    async def runtime(self, project: Path | None = None, **kw: Any) -> AsyncIterator[CoreRuntime]:
        rt = CoreRuntime.open(project_root=project, paths=self.paths, env=self.env, **kw)
        try:
            await rt.start()
            yield rt
        finally:
            await rt.close()

    def cli_env(self) -> dict[str, str]:
        return dict(self.env)
