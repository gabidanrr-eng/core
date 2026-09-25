"""Command execution tools. Commands are risk-analyzed by the policy engine before running;
raw shell interpretation is only used when explicitly requested (``use_shell``)."""

from __future__ import annotations

import os
import shlex

from pydantic import Field

from coremain.errors import ToolError
from coremain.security.commands import analyze_command
from coremain.security.env import build_subprocess_env
from coremain.security.policy import Capability, PolicyRequest
from coremain.tools.base import SideEffect, Tool, ToolContext, ToolInput, ToolResult
from coremain.tools.fs import guard
from coremain.verify.testparse import parse_test_output


def subprocess_env(ctx: ToolContext) -> dict[str, str]:
    return build_subprocess_env(os.environ, passthrough=ctx.services.config.permissions.env_passthrough, extra=ctx.extra_env)


class RunCommand(Tool):
    name = "run_command"
    description = ("Run a command in the workspace and return exit code and output. By default the command is split into argv "
                   "without a shell; set use_shell=true only when you need pipes, redirection or shell syntax. Commands are "
                   "risk-checked by policy; network, installs and destructive operations may require approval.")
    capability = Capability.EXEC
    side_effect = SideEffect.LOCAL
    read_only = False
    timeout_s = 3600.0

    class Input(ToolInput):
        command: str = Field(min_length=1)
        use_shell: bool = False
        cwd: str = "."
        timeout_s: int = Field(120, ge=1, le=3600)

    def target(self, args: Input) -> str:
        return args.command

    def summarize(self, args: Input) -> str:
        return f"$ {args.command}" + (f"  (in {args.cwd})" if args.cwd not in (".", "") else "")

    def effective_timeout(self, args: Input) -> float:
        return float(args.timeout_s) + 15.0

    def policy_request(self, ctx: ToolContext, args: Input) -> PolicyRequest | None:
        analysis = analyze_command(args.command, workspace=ctx.workspace.path)
        return PolicyRequest(capability=Capability.EXEC, target=args.command, workspace_root=ctx.workspace.path,
                             workspace_isolated=ctx.workspace.isolated, analysis=analysis, project_id=ctx.project_id,
                             session_id=ctx.session_id, task_id=ctx.task_id, tool=self.name)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        cwd, rel_cwd = guard(ctx, args.cwd)
        cfg = ctx.services.config.exec
        timeout = min(float(args.timeout_s), cfg.max_timeout_s)
        argv = None
        shell_command = None
        if args.use_shell:
            shell_command = args.command
        else:
            try:
                argv = shlex.split(args.command)
            except ValueError as exc:
                raise ToolError(f"cannot parse command: {exc}; set use_shell=true for shell syntax", error_class="invalid_command") from exc
            if any(tok in {"|", "&&", "||", ";", ">", ">>", "<"} for tok in argv):
                raise ToolError("command contains shell operators; set use_shell=true to run it through a shell",
                                error_class="needs_shell")
        events = ctx.services.events

        def on_output(stream: str, text: str) -> None:
            events.ephemeral("tool.output", project_id=ctx.project_id, session_id=ctx.session_id, task_id=ctx.task_id,
                             attempt_id=ctx.attempt_id, data={"stream": stream, "text": text[-4000:]})

        result = await ctx.services.processes.run(
            argv, shell_command=shell_command, cwd=cwd, env=subprocess_env(ctx), timeout_s=timeout, cancel=ctx.cancel.child(),
            on_output=on_output, task_id=ctx.task_id, attempt_id=ctx.attempt_id,
        )
        artifact_id = None
        if result.truncated and result.spool_path is not None:
            art = ctx.services.artifacts.put_text(result.spool_path.read_text(encoding="utf-8", errors="replace"), kind="command_output",
                                                  name=args.command[:80], project_id=ctx.project_id, task_id=ctx.task_id,
                                                  attempt_id=ctx.attempt_id)
            artifact_id = art.id
            result.spool_path.unlink(missing_ok=True)
        analysis = analyze_command(args.command, workspace=ctx.workspace.path)
        record = {"command": args.command, "cwd": rel_cwd, "exit_code": result.exit_code, "status": result.status,
                  "duration_s": round(result.duration_s, 2), "risk": analysis.max_risk.value}
        ctx.commands_run.append(record)
        summary = parse_test_output(result.stdout + "\n" + result.stderr) if "test" in analysis.max_risk.value else None
        head = f"exit code {result.exit_code}" if result.status == "exited" else f"{result.status} after {timeout:.0f}s"
        if summary:
            head += f" — {summary.line()}"
        body = result.combined(20_000) or "(no output)"
        notes = ("\n[" + "; ".join(result.notes) + "]") if result.notes else ""
        return ToolResult(result.ok, f"{head}\n{body}{notes}", status="ok" if result.status == "exited" else result.status,
                          error_class=None if result.ok else ("timeout" if result.status == "timeout" else "nonzero_exit"),
                          data=record, artifact_id=artifact_id)


class RunTests(Tool):
    name = "run_tests"
    description = ("Run the project's test command (from the project profile or verification config), optionally narrowed to a "
                   "path or test selector. Use this instead of guessing the test command.")
    capability = Capability.EXEC
    side_effect = SideEffect.LOCAL
    read_only = False
    timeout_s = 3600.0

    class Input(ToolInput):
        target: str | None = Field(None, description="Optional test path or selector appended to the command")
        timeout_s: int = Field(600, ge=5, le=3600)

    def _command(self, ctx: ToolContext, args: Input) -> str:
        cmd = ctx.services.config.verification.commands.get("test") or (ctx.services.profile.get("commands") or {}).get("test")
        if not cmd:
            raise ToolError("no test command is known for this project; inspect the repository or use run_command",
                            error_class="no_test_command")
        if args.target:
            cmd = f"{cmd} {shlex.quote(args.target)}"
        return str(cmd)

    def target(self, args: Input) -> str:
        return args.target or "(project tests)"

    def effective_timeout(self, args: Input) -> float:
        return float(args.timeout_s) + 15.0

    def policy_request(self, ctx: ToolContext, args: Input) -> PolicyRequest | None:
        try:
            cmd = self._command(ctx, args)
        except ToolError:
            return None
        return PolicyRequest(capability=Capability.EXEC, target=cmd, workspace_root=ctx.workspace.path,
                             workspace_isolated=ctx.workspace.isolated, analysis=analyze_command(cmd, workspace=ctx.workspace.path),
                             project_id=ctx.project_id, session_id=ctx.session_id, task_id=ctx.task_id, tool=self.name)

    async def run(self, ctx: ToolContext, args: Input) -> ToolResult:
        cmd = self._command(ctx, args)
        inner = RunCommand()
        return await inner.run(ctx, RunCommand.Input(command=cmd, use_shell=True, timeout_s=args.timeout_s))


SHELL_TOOLS: list[type[Tool]] = [RunCommand, RunTests]
