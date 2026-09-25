"""LSPManager and the lsp_* tools end to end.

A real CoreRuntime is opened in a temporary home, the fake language server is configured through
the config file, and every tool call goes through ToolExecutor (argument validation, policy,
path guards, output finalization) exactly as agent tool calls do.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.fixtures.lsp_support import (
    FAKE_SERVER,
    PROJECT_FILES,
    events,
    fake_command,
    lsp_toml,
    read_log,
    received,
    wait_for_log,
    wait_gone,
)
from tests.helpers import Harness, make_repo, write_files

from coremain.intel.lsp import LSPDefinition, LSPHover, LSPManager
from coremain.providers.types import ToolCall
from coremain.runtime.app import CoreRuntime
from coremain.runtime.cancel import CancelToken
from coremain.security.paths import PathViolation
from coremain.store.db import Database
from coremain.tools.base import ToolContext, ToolResult
from coremain.tools.executor import ToolExecutor
from coremain.workspaces.manager import Workspace

LSP_NAMES = {"lsp_definition", "lsp_references", "lsp_hover", "lsp_diagnostics", "lsp_document_symbols"}
ROLES = ("planner", "implementer", "corrector", "debugger", "researcher", "reviewer")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "proj", PROJECT_FILES)


@pytest.fixture
def hermetic(harness: Harness, tmp_path: Path) -> Harness:
    """No real language servers on PATH, so built-in discovery is deterministic on any machine."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    harness.env["PATH"] = str(empty)
    return harness


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "lsp.log"


def manager(rt: CoreRuntime, **settings: float) -> LSPManager:
    mgr = rt.extension("lsp")
    assert isinstance(mgr, LSPManager)
    for key, value in {"request_timeout_s": 5.0, "startup_timeout_s": 10.0, **settings}.items():
        setattr(mgr.settings, key, value)
    return mgr


def context(rt: CoreRuntime, workspace: Workspace | None = None, role: str = "implementer") -> ToolContext:
    project = rt.require_project()
    return ToolContext(
        project_id=project.id,
        session_id=None,
        task_id=None,
        attempt_id=None,
        workspace=workspace or rt.workspaces.canonical(project),
        fence=None,
        cancel=CancelToken(),
        role=role,
        services=rt.tool_services(project, {}),
    )


async def call(mgr: LSPManager, ctx: ToolContext, name: str, **arguments: Any) -> ToolResult:
    tools = {t.name: t for t in mgr.tools_for_role(ctx.role)}
    return await ToolExecutor(tools).execute(ToolCall(id=f"call-{name}", name=name, arguments=arguments), ctx)


def process_rows(db: Any) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in db.query(
            "SELECT pid, status, exit_code, runtime_id FROM processes WHERE command LIKE ? ORDER BY started_at",
            (f"%{FAKE_SERVER.name}%",),
        )
    ]


# ---------------------------------------------------------------------------- happy path
async def test_tools_answer_through_executor_for_every_role(hermetic: Harness, repo: Path, log: Path) -> None:
    hermetic.config(lsp_toml(fake_command(log)))
    async with hermetic.runtime(repo) as rt:
        mgr = manager(rt)
        for role in ROLES:
            assert {t.name for t in mgr.tools_for_role(role)} == LSP_NAMES
            assert all(t.read_only for t in mgr.tools_for_role(role))
        assert {t.name for t in rt.extra_tools("reviewer")} >= LSP_NAMES
        assert mgr.tools_for_role("unknown-role") == []
        ctx = context(rt)

        result = await call(mgr, ctx, "lsp_definition", path="main.py", line=3, character=17)
        assert result.ok, result.content
        assert result.content.splitlines() == [
            "definition of 'add' at main.py:3:17 via fake: 1 location(s)",
            "calc.py:1:5  def add(a, b):",
        ]

        result = await call(mgr, ctx, "lsp_references", path="calc.py", symbol="add")
        assert result.ok, result.content
        lines = result.content.splitlines()
        assert lines[0].startswith("references to 'add' at calc.py:1:5 via fake: 3 in 2 file(s)")
        assert "[definition of 'add' in calc.py]" in lines[0]
        # Column 17 is the character column even though the server speaks UTF-16 (offset 17 there).
        assert lines[1:] == [
            "calc.py:1:5  def add(a, b):",
            "main.py:1:30  from calc import Calculator, add",
            'main.py:3:17  s = "é🙂"; print(add(1, 2))',
        ]
        assert result.data is not None and result.data["count"] == 3

        result = await call(
            mgr, ctx, "lsp_references", path="calc.py", symbol="add", include_declaration=False
        )
        lines = result.content.splitlines()
        assert "2 in 1 file(s)" in lines[0]
        assert [line.split("  ")[0] for line in lines[1:]] == ["main.py:1:30", "main.py:3:17"]

        result = await call(mgr, ctx, "lsp_hover", path="main.py", line=3, symbol="add")
        assert result.ok and "def add(a, b)" in result.content and "Fake documentation" in result.content

        result = await call(mgr, ctx, "lsp_document_symbols", path="calc.py")
        assert result.ok, result.content
        assert result.content.splitlines() == [
            "calc.py: 3 symbol(s) via fake",
            "1:5 function add — (a, b)",
            "6:7 class Calculator",
            "  7:9 method total — (self, items)",
        ]

        result = await call(mgr, ctx, "lsp_diagnostics", path="main.py")
        assert result.ok, result.content
        assert result.content.splitlines() == [
            "main.py: 1 error from fake (push)",
            "main.py:4:9 error [fake E001] undefined name 'undefined_name'",
        ]
        assert result.data is not None and result.data["received"] and result.data["fresh"]

        status = mgr.status()
        assert status["enabled"] and status["tools_offered"]
        assert status["project_languages"]["python"] == {"server": "fake"}
        fake = next(s for s in status["servers"] if s["name"] == "fake")
        assert fake["available"] and fake["source"] == "config"
        assert [c["state"] for c in fake["clients"]] == ["ready"]
        assert fake["clients"][0]["root"] == str(repo.resolve())
        pyright = next(s for s in status["servers"] if s["name"] == "pyright")
        assert not pyright["available"] and pyright["reason"] == "python handled by configured server(s)"
        gopls = next(s for s in status["servers"] if s["name"] == "gopls")
        assert gopls["reason"] == "not installed ('gopls' not found on PATH)"
        # Tool calls are recorded like every other tool call.
        rows = rt.db.query("SELECT tool, status FROM tool_calls WHERE tool LIKE 'lsp_%'")
        assert {r["tool"] for r in rows} == LSP_NAMES and {r["status"] for r in rows} == {"ok"}


async def test_diagnostics_follow_disk_changes(hermetic: Harness, repo: Path, log: Path) -> None:
    hermetic.config(lsp_toml(fake_command(log)))
    async with hermetic.runtime(repo) as rt:
        mgr, ctx = manager(rt), context(rt)
        assert "1 error" in (await call(mgr, ctx, "lsp_diagnostics", path="main.py")).content
        (repo / "main.py").write_text("from calc import add\n\nvalue = add(1, 2)\n", encoding="utf-8")
        result = await call(mgr, ctx, "lsp_diagnostics", path="main.py")
        assert result.content == "main.py: no diagnostics from fake (push)"
        changes = received(log, "textDocument/didChange")
        assert [c["params"]["textDocument"]["version"] for c in changes] == [2]


async def test_pull_diagnostics_reuse_unchanged_reports(hermetic: Harness, repo: Path, log: Path) -> None:
    hermetic.config(lsp_toml(fake_command(log, "--pull-diagnostics", "--no-publish")))
    async with hermetic.runtime(repo) as rt:
        mgr, ctx = manager(rt), context(rt)
        for _ in range(2):
            result = await call(mgr, ctx, "lsp_diagnostics", path="main.py", wait_seconds=0)
            assert result.content.splitlines() == [
                "main.py: 1 error from fake (pull)",
                "main.py:4:9 error [fake E001] undefined name 'undefined_name'",
            ]
        pulls = received(log, "textDocument/diagnostic")
        assert len(pulls) == 2
        assert pulls[0]["params"]["identifier"] == "fake"
        assert "previousResultId" not in pulls[0]["params"]
        assert pulls[1]["params"]["previousResultId"]  # answered with kind=unchanged


async def test_missing_diagnostics_are_not_reported_as_clean(
    hermetic: Harness, repo: Path, log: Path
) -> None:
    hermetic.config(lsp_toml(fake_command(log, "--no-publish")))
    async with hermetic.runtime(repo) as rt:
        mgr, ctx = manager(rt), context(rt)
        result = await call(mgr, ctx, "lsp_diagnostics", path="main.py", wait_seconds=0.3)
        assert result.ok
        assert "published no diagnostics" in result.content
        assert "not evidence that the file is clean" in result.content
        assert result.data is not None and result.data["received"] is False


async def test_encodings_location_links_and_flat_symbols(hermetic: Harness, repo: Path, log: Path) -> None:
    flags = ("--position-encoding", "utf-8", "--location-links", "--flat-symbols")
    hermetic.config(lsp_toml(fake_command(log, *flags)))
    async with hermetic.runtime(repo) as rt:
        mgr, ctx = manager(rt), context(rt)
        result = await call(mgr, ctx, "lsp_definition", path="main.py", line=3, symbol="add")
        assert result.content.splitlines()[1] == "calc.py:1:5  def add(a, b):"
        result = await call(mgr, ctx, "lsp_references", path="main.py", line=3, character=18)
        assert 'main.py:3:17  s = "é🙂"; print(add(1, 2))' in result.content.splitlines()
        # Column 18 is the second letter of `add`: 17 code points before it, which is 21 UTF-8 bytes
        # because é takes 2 bytes and 🙂 takes 4.
        assert received(log, "textDocument/references")[0]["params"]["position"] == {
            "line": 2,
            "character": 21,
        }
        result = await call(mgr, ctx, "lsp_document_symbols", path="calc.py")
        assert "7:9 method total (in Calculator)" in result.content.splitlines()
        assert events(log, "initialize")[0]["encoding"] == "utf-8"


async def test_builtin_server_is_used_only_when_on_path(
    harness: Harness, repo: Path, log: Path, tmp_path: Path
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_file = tmp_path / "argv.txt"
    wrapper = bindir / "pyright-langserver"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(argv_file))}\n"
        f'exec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_SERVER))} --log {shlex.quote(str(log))} "$@"\n'
    )
    wrapper.chmod(0o755)
    harness.env["PATH"] = str(bindir)
    async with harness.runtime(repo) as rt:
        mgr = manager(rt)
        status = mgr.status()
        pyright = next(s for s in status["servers"] if s["name"] == "pyright")
        assert (
            pyright["available"] and pyright["executable"] == str(wrapper) and pyright["source"] == "builtin"
        )
        basedpyright = next(s for s in status["servers"] if s["name"] == "basedpyright")
        assert not basedpyright["available"]
        assert status["project_languages"] == {"python": {"server": "pyright"}}
        result = await call(mgr, context(rt), "lsp_definition", path="main.py", line=3, character=17)
        assert result.ok and "calc.py:1:5" in result.content
        init = received(log, "initialize")[0]
        assert init["params"]["rootUri"] == repo.resolve().as_uri()
        assert init["params"]["processId"] == os.getpid()
        assert init["params"]["capabilities"]["workspace"]["applyEdit"] is False
    assert argv_file.read_text().split() == ["--stdio"]


# ------------------------------------------------------------------------------ failures
async def test_timeout_is_classified_and_server_stays_usable(
    hermetic: Harness, repo: Path, log: Path
) -> None:
    hermetic.config(lsp_toml(fake_command(log, "--hang-on", "textDocument/hover")))
    async with hermetic.runtime(repo) as rt:
        mgr, ctx = manager(rt, request_timeout_s=0.4), context(rt)
        result = await call(mgr, ctx, "lsp_hover", path="main.py", line=3, character=17)
        assert not result.ok and result.error_class == "lsp_timeout"
        assert "did not answer textDocument/hover within 0.4s" in result.content
        assert "hint: fall back to find_symbol" in result.content
        await wait_for_log(log, lambda es: any(e.get("event") == "cancelled" for e in es))
        assert (await call(mgr, ctx, "lsp_definition", path="main.py", line=3, character=17)).ok
        assert len(received(log, "initialize")) == 1


async def test_crash_is_reported_then_the_next_call_restarts(
    hermetic: Harness, repo: Path, log: Path, tmp_path: Path
) -> None:
    flags = (
        "--crash-on",
        "textDocument/definition",
        "--crash-times",
        "1",
        "--state-file",
        str(tmp_path / "n"),
    )
    hermetic.config(lsp_toml(fake_command(log, *flags)))
    async with hermetic.runtime(repo) as rt:
        mgr, ctx = manager(rt, restart_backoff_s=0.01), context(rt)
        result = await call(mgr, ctx, "lsp_definition", path="main.py", line=3, character=17)
        assert not result.ok and result.error_class == "lsp_crashed"
        assert (
            "exited with code 3" in result.content and "crashing on textDocument/definition" in result.content
        )
        result = await call(mgr, ctx, "lsp_definition", path="main.py", line=3, character=17)
        assert result.ok and "calc.py:1:5" in result.content
        client = mgr.status()["servers"][0]["clients"][0]
        assert (client["crashes"], client["restarts"], client["state"]) == (1, 1, "ready")
        rows = process_rows(rt.db)
        assert [(r["status"], r["exit_code"]) for r in rows] == [("exited", 3), ("running", None)]
        assert all(r["runtime_id"] == rt.runtime_id for r in rows)
        kinds = [
            r["kind"] for r in rt.db.query("SELECT kind FROM events WHERE kind LIKE 'lsp.%' ORDER BY seq")
        ]
        assert kinds == ["lsp.started", "lsp.crashed", "lsp.started"]


async def test_graceful_shutdown_when_runtime_closes(hermetic: Harness, repo: Path, log: Path) -> None:
    hermetic.config(lsp_toml(fake_command(log)))
    async with hermetic.runtime(repo) as rt:
        mgr = manager(rt)
        assert (await call(mgr, context(rt), "lsp_document_symbols", path="calc.py")).ok
        pid = mgr.status()["servers"][0]["clients"][0]["pid"]
    assert await wait_gone(pid)
    methods = [e["in"].get("method") for e in read_log(log) if "in" in e]
    assert methods[-2:] == ["shutdown", "exit"]
    db = Database(hermetic.paths.db_path)
    try:
        assert [(r["status"], r["exit_code"]) for r in process_rows(db)] == [("exited", 0)]
    finally:
        db.close()


async def test_kill_fallback_leaves_no_orphans(hermetic: Harness, repo: Path, log: Path) -> None:
    hermetic.config(lsp_toml(fake_command(log, "--ignore-exit", "--spawn-child")))
    async with hermetic.runtime(repo) as rt:
        mgr = manager(rt, exit_wait_s=0.2, kill_grace_s=0.2)
        assert (await call(mgr, context(rt), "lsp_hover", path="main.py", line=3, character=17)).ok
        pid = mgr.status()["servers"][0]["clients"][0]["pid"]
        child = events(log, "child")[0]["child_pid"]
        await mgr.aclose()
        assert await wait_gone(pid) and await wait_gone(child)
        assert [(r["status"], r["exit_code"]) for r in process_rows(rt.db)] == [("killed", -9)]
        assert mgr.tools_for_role("implementer") == []


async def test_disabled_by_configuration(hermetic: Harness, repo: Path, log: Path) -> None:
    hermetic.config(lsp_toml(fake_command(log), enabled=False))
    async with hermetic.runtime(repo) as rt:
        mgr = manager(rt)
        assert all(mgr.tools_for_role(role) == [] for role in ROLES)
        assert not LSP_NAMES & {t.name for t in rt.extra_tools("implementer")}
        status = mgr.status()
        assert status["enabled"] is False and status["tools_offered"] is False
        assert status["reason"] == "disabled by configuration (intel.lsp_enabled = false)"
        assert status["project_languages"]["python"]["server"] is None
        # Even a directly constructed tool refuses instead of starting a server.
        tool = LSPDefinition(mgr)
        result = await ToolExecutor({tool.name: tool}).execute(
            ToolCall(id="c", name=tool.name, arguments={"path": "main.py", "line": 3, "character": 17}),
            context(rt),
        )
        assert not result.ok and result.error_class == "lsp_unavailable"
    assert not log.exists()


async def test_missing_configured_server_shadows_builtins_and_hides_tools(
    hermetic: Harness, repo: Path
) -> None:
    hermetic.config(lsp_toml(["definitely-not-a-language-server-xyz", "--stdio"]))
    async with hermetic.runtime(repo) as rt:
        mgr = manager(rt)
        assert mgr.tools_for_role("implementer") == []
        status = mgr.status()
        fake = next(s for s in status["servers"] if s["name"] == "fake")
        assert fake["reason"] == "executable 'definitely-not-a-language-server-xyz' not found on PATH"
        python = status["project_languages"]["python"]
        assert python["server"] is None and "definitely-not-a-language-server-xyz" in python["reason"]
        pylsp = next(s for s in status["servers"] if s["name"] == "pylsp")
        assert pylsp["reason"] == "python handled by configured server(s)"


async def test_no_server_for_the_project_languages(hermetic: Harness, repo: Path, log: Path) -> None:
    hermetic.config(lsp_toml(fake_command(log), languages=["rust"]))
    async with hermetic.runtime(repo) as rt:
        mgr = manager(rt)
        assert mgr.tools_for_role("reviewer") == []
        python = mgr.status()["project_languages"]["python"]
        assert python["server"] is None and "pyright: not installed" in python["reason"]
        tool = LSPHover(mgr)
        result = await ToolExecutor({tool.name: tool}).execute(
            ToolCall(id="c", name=tool.name, arguments={"path": "main.py", "symbol": "add"}), context(rt)
        )
        assert not result.ok and result.error_class == "lsp_unavailable"
        assert "no language server available for python files" in result.content


async def test_path_escapes_and_bad_arguments_are_refused(
    hermetic: Harness, repo: Path, log: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("def secret():\n    pass\n", encoding="utf-8")
    (repo / "link.py").symlink_to(outside)
    write_files(repo, {".env": "TOKEN=abc\n", "pkg/__init__.py": ""})
    hermetic.config(lsp_toml(fake_command(log)))
    async with hermetic.runtime(repo) as rt:
        mgr, ctx = manager(rt), context(rt)
        for path in ("../outside.py", str(outside), "link.py", ".env", ".git/config"):
            result = await call(mgr, ctx, "lsp_definition", path=path, line=1, character=5)
            assert not result.ok and result.status == "denied", (path, result.content)
        # The tool's own guard holds even without the executor's policy check.
        tool = next(t for t in mgr.tools_for_role("implementer") if t.name == "lsp_hover")
        with pytest.raises(PathViolation):
            await tool.run(ctx, tool.Input(path="../outside.py", line=1, character=5))
        expectations = [
            ({"path": "main.py"}, "invalid_arguments"),
            ({"path": "main.py", "line": 1}, "invalid_arguments"),
            ({"path": "main.py", "line": 0, "character": 1}, "invalid_arguments"),
            ({"path": "main.py", "line": 99, "character": 1}, "invalid_position"),
            ({"path": "main.py", "symbol": "nothing_like_this"}, "symbol_not_found"),
            ({"path": "missing.py", "line": 1, "character": 1}, "not_found"),
            ({"path": "pkg", "line": 1, "character": 1}, "not_a_file"),
            ({"path": "pyproject.toml", "line": 1, "character": 1}, "lsp_unavailable"),
        ]
        for arguments, error_class in expectations:
            result = await call(mgr, ctx, "lsp_hover", **arguments)
            assert not result.ok and result.error_class == error_class, (arguments, result.content)
        # Nothing was ever sent for the refused files.
        opened = {
            Path(m["params"]["textDocument"]["uri"]).name for m in received(log, "textDocument/didOpen")
        }
        assert opened <= {"main.py"}


async def test_output_is_bounded_and_outside_files_are_not_read(
    hermetic: Harness, repo: Path, log: Path, tmp_path: Path
) -> None:
    external = tmp_path / "stubs" / "builtins.pyi"
    external.parent.mkdir()
    external.write_text("SECRET_PREVIEW_TEXT = 1\n", encoding="utf-8")
    hermetic.config(lsp_toml(fake_command(log, "--external-def", str(external))))
    async with hermetic.runtime(repo) as rt:
        mgr, ctx = manager(rt), context(rt)
        result = await call(mgr, ctx, "lsp_references", path="calc.py", symbol="add", max_results=1)
        lines = result.content.splitlines()
        assert len(lines) == 3 and lines[-1] == "… 2 more not shown (raise max_results, up to 500)"
        result = await call(mgr, ctx, "lsp_definition", path="main.py", line=3, symbol="print")
        assert result.ok
        assert f"{external}:1:1  (outside the workspace; not read)" in result.content.splitlines()
        assert "SECRET_PREVIEW_TEXT" not in result.content


async def test_one_server_per_workspace_root_and_reaping(
    hermetic: Harness, repo: Path, log: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "ws-copy"
    write_files(copy, PROJECT_FILES)
    hermetic.config(lsp_toml(fake_command(log)))
    async with hermetic.runtime(repo) as rt:
        mgr = manager(rt)
        canonical = context(rt)
        project = rt.require_project()
        isolated = Workspace("ws-copy", project.id, "copy", copy, None, None, "active", None, None, {}, 0.0)
        assert (await call(mgr, canonical, "lsp_definition", path="main.py", line=3, character=17)).ok
        result = await call(
            mgr, context(rt, isolated), "lsp_definition", path="main.py", line=3, character=17
        )
        assert result.ok and result.content.splitlines()[1] == "calc.py:1:5  def add(a, b):"
        clients = {c["root"]: c["pid"] for c in mgr.status()["servers"][0]["clients"]}
        assert set(clients) == {str(repo.resolve()), str(copy.resolve())}
        assert len(set(clients.values())) == 2
        # The isolated workspace disappears; its server is stopped on the next use of the manager.
        shutil.rmtree(copy)
        assert (await call(mgr, canonical, "lsp_hover", path="main.py", line=3, character=17)).ok
        assert await wait_gone(clients[str(copy.resolve())])
        assert [c["root"] for c in mgr.status()["servers"][0]["clients"]] == [str(repo.resolve())]


async def test_client_cap_stops_least_recently_used_server(
    hermetic: Harness, repo: Path, log: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "ws-copy"
    write_files(copy, PROJECT_FILES)
    hermetic.config(lsp_toml(fake_command(log)))
    async with hermetic.runtime(repo) as rt:
        mgr = manager(rt, max_clients=1)
        project = rt.require_project()
        isolated = Workspace("ws-copy", project.id, "copy", copy, None, None, "active", None, None, {}, 0.0)
        assert (await call(mgr, context(rt), "lsp_hover", path="main.py", line=3, character=17)).ok
        first = mgr.status()["servers"][0]["clients"][0]["pid"]
        # Reusing the running client does not evict anything.
        assert (await call(mgr, context(rt), "lsp_hover", path="calc.py", line=1, character=5)).ok
        assert mgr.status()["servers"][0]["clients"][0]["pid"] == first
        assert (await call(mgr, context(rt, isolated), "lsp_hover", path="main.py", line=3, character=17)).ok
        assert await wait_gone(first)
        clients = mgr.status()["servers"][0]["clients"]
        assert [c["root"] for c in clients] == [str(copy.resolve())]
