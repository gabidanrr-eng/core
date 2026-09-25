"""`core config`, `core permissions`, `core providers`, `core models`."""

from __future__ import annotations

import getpass
import sys
from pathlib import Path
from typing import Any

import click

from coremain.cli.common import CLIContext, CommandExit, run_async, runtime_command
from coremain.cli.output import ago
from coremain.config.edit import parse_value, set_in, unset_in, update_toml
from coremain.config.loader import load_config, read_toml
from coremain.errors import ConfigError, NotFoundError, UsageError
from coremain.paths import ProjectPaths, resolve_core_paths
from coremain.util.text import one_line


def _target_file(ctx: CLIContext, project: bool, local: bool) -> Path:
    if project or local:
        root = ctx.project_root()
        if root is None:
            raise UsageError("--project/--local require a project")
        pp = ProjectPaths(root)
        return pp.local_config_file if local else pp.config_file
    return resolve_core_paths(ctx.env).config_file


def _validated_update(ctx: CLIContext, path: Path, mutate: Any) -> None:
    """Apply a config edit, re-validate the effective configuration, and roll back on error."""
    original = path.read_bytes() if path.exists() else None
    update_toml(path, mutate)
    try:
        load_config(resolve_core_paths(ctx.env), ctx.project_root(), env=ctx.env, project_trusted=True)
    except ConfigError:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)
        raise


# ============================================================================ config
@click.group()
def config() -> None:
    """View and edit layered configuration."""


@config.command("show")
@click.option("--origins", is_flag=True, help="Show which layer set each value.")
@runtime_command(require_project=False)
async def config_show(ctx: CLIContext, rt: Any, origins: bool) -> int:
    data = rt.config.model_dump(mode="json")
    payload = {
        "config": data,
        "origins": rt.effective.origins if origins else None,
        "warnings": rt.effective.warnings,
        "layers": [
            {"name": layer.name, "path": str(layer.path) if layer.path else None, "trusted": layer.trusted}
            for layer in rt.effective.layers
        ],
    }

    def human(c: Any) -> None:
        import tomli_w

        c.print(
            "[bold]Layers:[/] " + " < ".join(["defaults", *[layer.name for layer in rt.effective.layers]])
        )
        c.print(tomli_w.dumps({k: v for k, v in data.items() if v not in ({}, [], None)}), markup=False)
        if origins:
            for key, origin in sorted(rt.effective.origins.items()):
                c.print(f"[dim]{key} ← {origin}[/]")
        for w in rt.effective.warnings:
            c.print(f"[yellow]warning:[/] {w}")

    ctx.output.data(payload, human)
    return 0


@config.command("get")
@click.argument("key")
@runtime_command(require_project=False)
async def config_get(ctx: CLIContext, rt: Any, key: str) -> int:
    value: Any = rt.config.model_dump(mode="json")
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise NotFoundError(f"no configuration key '{key}'")
        value = value[part]
    ctx.output.data(
        {"key": key, "value": value, "origin": rt.effective.origin_of(key)}, lambda c: c.print(repr(value))
    )
    return 0


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--project", "to_project", is_flag=True, help="Write to .core/config.toml.")
@click.option("--local", "to_local", is_flag=True, help="Write to .core/config.local.toml (not committed).")
@click.pass_obj
def config_set(ctx: CLIContext, key: str, value: str, to_project: bool, to_local: bool) -> None:
    """Set KEY to VALUE (TOML literal or string); validated before it is kept."""
    path = _target_file(ctx, to_project, to_local)
    parsed = parse_value(value)
    try:
        _validated_update(ctx, path, lambda d: set_in(d, key, parsed))
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data({"key": key, "value": parsed, "file": str(path)}, f"set {key} = {parsed!r} in {path}")


@config.command("unset")
@click.argument("key")
@click.option("--project", "to_project", is_flag=True)
@click.option("--local", "to_local", is_flag=True)
@click.pass_obj
def config_unset(ctx: CLIContext, key: str, to_project: bool, to_local: bool) -> None:
    path = _target_file(ctx, to_project, to_local)
    removed: list[bool] = []
    try:
        _validated_update(ctx, path, lambda d: removed.append(unset_in(d, key)))
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data(
        {"key": key, "removed": bool(removed and removed[0])},
        f"unset {key}" if removed and removed[0] else f"{key} was not set in {path}",
    )


@config.command("validate")
@click.pass_obj
def config_validate(ctx: CLIContext) -> None:
    """Validate every configuration layer."""
    try:
        eff = load_config(resolve_core_paths(ctx.env), ctx.project_root(), env=ctx.env, project_trusted=True)
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data(
        {"valid": True, "warnings": eff.warnings, "layers": [layer.name for layer in eff.layers]},
        lambda c: c.print(
            "[green]configuration is valid[/]" + "".join(f"\n[yellow]warning:[/] {w}" for w in eff.warnings)
        ),
    )


@config.command("path")
@click.pass_obj
def config_path(ctx: CLIContext) -> None:
    """Show configuration and state locations."""
    paths = resolve_core_paths(ctx.env)
    root = ctx.project_root()
    data = {
        "user_config": str(paths.config_file),
        "credentials": str(paths.credentials_file),
        "data_dir": str(paths.data_dir),
        "database": str(paths.db_path),
        "artifacts": str(paths.artifacts_dir),
        "workspaces": str(paths.workspaces_dir),
        "cache": str(paths.cache_dir),
        "logs": str(paths.logs_dir),
        "project_config": str(ProjectPaths(root).config_file) if root else None,
    }
    ctx.output.data(data, lambda c: [c.print(f"{k:<15} {v}") for k, v in data.items()])


# ============================================================================ permissions
@click.group()
def permissions() -> None:
    """Effective permissions, grants and policy what-if checks."""


@permissions.command("show")
@runtime_command(require_project=False)
async def permissions_show(ctx: CLIContext, rt: Any) -> int:
    grants = rt.policy.active_grants(rt.project.id if rt.project else None)
    data = {**rt.policy.describe(), "grants": grants}

    def human(c: Any) -> None:
        c.print(f"[bold]Profile[/] {data['profile']}" + (" (offline)" if data["offline"] else ""))
        c.print(f"non-interactive 'ask' → {data['non_interactive_ask']}")
        for r in data["rules"]:
            c.print(f"  rule {r['capability']} {r['pattern']} → {r['decision']}")
        for g in grants:
            scope = (
                "task " + g["task_id"]
                if g["task_id"]
                else ("session " + g["session_id"] if g["session_id"] else "project")
            )
            c.print(
                f"  grant {g['id']} {g['capability']} '{g['pattern']}' → {g['decision']} ({scope}; by {g['granted_by']})"
            )
        c.print(
            "[dim]Invariants (not configurable): no writes to Core Main config/state, no access outside the workspace, "
            "no secret files or secret-revealing commands without an explicit secrets.read grant, no privileged commands by default.[/]"
        )

    ctx.output.data(data, human)
    return 0


@permissions.command("grant")
@click.argument("capability")
@click.option(
    "--pattern",
    default="*",
    show_default=True,
    help="Glob over the target (path, command, domain, server:tool).",
)
@click.option("--session", "session_id", default=None)
@click.option("--task", "task_id", default=None)
@click.option("--global", "global_scope", is_flag=True, help="Apply to every project.")
@click.option("--ttl", type=float, default=None, help="Expire after N seconds.")
@click.option("--uses", type=int, default=None, help="Expire after N uses.")
@click.option("--deny", is_flag=True, help="Create a deny grant instead.")
@click.option("--reason", default=None)
@runtime_command(require_project=False)
async def permissions_grant(
    ctx: CLIContext,
    rt: Any,
    capability: str,
    pattern: str,
    session_id: str | None,
    task_id: str | None,
    global_scope: bool,
    ttl: float | None,
    uses: int | None,
    deny: bool,
    reason: str | None,
) -> int:
    project_id = None if global_scope or rt.project is None else rt.project.id
    grant_id = rt.policy.grant(
        capability,
        pattern=pattern,
        decision="deny" if deny else "allow",
        granted_by="user",
        project_id=project_id,
        session_id=session_id,
        task_id=task_id,
        ttl_s=ttl,
        uses=uses,
        reason=reason,
    )
    rt.events.emit(
        "permission.granted",
        project_id=project_id,
        actor="user",
        data={
            "grant_id": grant_id,
            "capability": capability,
            "pattern": pattern,
            "decision": "deny" if deny else "allow",
        },
    )
    ctx.output.data({"grant_id": grant_id}, f"created grant {grant_id}")
    return 0


@permissions.command("revoke")
@click.argument("grant_id")
@runtime_command(require_project=False)
async def permissions_revoke(ctx: CLIContext, rt: Any, grant_id: str) -> int:
    if not rt.policy.revoke(grant_id):
        raise NotFoundError(f"no active grant {grant_id}")
    ctx.output.data({"revoked": grant_id}, f"revoked {grant_id}")
    return 0


@permissions.command("check")
@click.argument("capability")
@click.argument("target")
@click.option("--canonical", is_flag=True, help="Evaluate as if in the canonical (non-isolated) workspace.")
@runtime_command(require_project=False)
async def permissions_check(ctx: CLIContext, rt: Any, capability: str, target: str, canonical: bool) -> int:
    """What would the policy decide for CAPABILITY on TARGET? (e.g. `exec.run "rm -rf /"`)"""
    from coremain.security.commands import analyze_command
    from coremain.security.policy import PolicyRequest

    root = rt.project_root
    analysis = analyze_command(target, workspace=root) if capability == "exec.run" else None
    decision = rt.policy.evaluate(
        PolicyRequest(
            capability=capability,
            target=target,
            workspace_root=root,
            workspace_isolated=not canonical,
            analysis=analysis,
            project_id=rt.project.id if rt.project else None,
        )
    )
    data = {
        **decision.to_dict(),
        "analysis": {"risks": sorted(r.value for r in analysis.risks), "reasons": analysis.reasons}
        if analysis
        else None,
    }
    style = {"allow": "green", "ask": "yellow", "deny": "red"}[decision.decision]
    ctx.output.data(
        data,
        lambda c: c.print(
            f"[{style}]{decision.decision}[/] — {decision.reason} [dim]({decision.source})[/]"
            + (f"\n[dim]risks: {', '.join(data['analysis']['risks'])}[/]" if analysis else "")
        ),
    )
    return 0


# ============================================================================ providers
@click.group()
def providers() -> None:
    """Model providers (identity, endpoint, credentials)."""


@providers.command("list")
@runtime_command(require_project=False)
async def providers_list(ctx: CLIContext, rt: Any) -> int:
    from coremain.providers.registry import PROVIDER_PRESETS

    rows = [
        {
            "id": pid,
            "kind": cfg.kind,
            "base_url": cfg.base_url,
            "enabled": cfg.enabled,
            "api_key": cfg.api_key,
            "credential": rt.registry.credential_status(pid).describe(),
        }
        for pid, cfg in rt.config.providers.items()
    ]

    def human(c: Any) -> None:
        if rows:
            ctx.output.table(
                ["id", "kind", "base_url", "credential ref", "status"],
                [
                    [r["id"], r["kind"], r["base_url"] or "", r["api_key"] or "", r["credential"]]
                    for r in rows
                ],
            )
        else:
            c.print("No providers configured.")
        c.print(
            "[dim]Presets: " + ", ".join(sorted(PROVIDER_PRESETS)) + "  (`core providers add <preset>`)[/]"
        )

    ctx.output.data({"providers": rows}, human)
    return 0


@providers.command("add")
@click.argument("name")
@click.option("--preset", default=None, help="Endpoint preset (defaults to NAME if it is a preset).")
@click.option("--kind", type=click.Choice(["openai_compatible", "anthropic", "scripted"]), default=None)
@click.option("--base-url", default=None)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help="Credential reference: env:VAR, store:NAME, file:PATH or command:CMD.",
)
@click.option("--script", default=None, help="Script path (scripted provider, for local development/tests).")
@click.option(
    "--project",
    "to_project",
    is_flag=True,
    help="Write to project config (requires `core trust` to take effect).",
)
@click.pass_obj
def providers_add(
    ctx: CLIContext,
    name: str,
    preset: str | None,
    kind: str | None,
    base_url: str | None,
    api_key: str | None,
    script: str | None,
    to_project: bool,
) -> None:
    from coremain.providers.registry import PROVIDER_PRESETS

    base = dict(PROVIDER_PRESETS.get(preset or name, {}))
    if preset and not base:
        raise CommandExit(
            int(
                ctx.output.error(
                    UsageError(f"unknown preset '{preset}'", hint=", ".join(sorted(PROVIDER_PRESETS)))
                )
            )
        )
    entry = {
        **base,
        **{
            k: v
            for k, v in {"kind": kind, "base_url": base_url, "api_key": api_key, "script": script}.items()
            if v
        },
    }
    if "kind" not in entry:
        raise CommandExit(int(ctx.output.error(UsageError("specify --kind (or use a preset)"))))
    path = _target_file(ctx, to_project, False)
    try:
        _validated_update(ctx, path, lambda d: set_in(d, f"providers.{name}", entry))
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data(
        {"provider": name, "config": entry, "file": str(path)},
        f"added provider [bold]{name}[/] ({entry['kind']}) to {path}. Next: register a model with "
        f"`core models add <key> --provider {name} --id <exact-model-id> --from-discovery`.",
    )


@providers.command("remove")
@click.argument("name")
@click.option("--project", "to_project", is_flag=True)
@click.pass_obj
def providers_remove(ctx: CLIContext, name: str, to_project: bool) -> None:
    path = _target_file(ctx, to_project, False)
    data = read_toml(path)
    dependents = [k for k, m in (data.get("models") or {}).items() if m.get("provider") == name]
    if dependents:
        raise CommandExit(
            int(
                ctx.output.error(
                    UsageError(
                        f"models depend on provider {name}: {', '.join(dependents)}; remove them first"
                    )
                )
            )
        )
    try:
        _validated_update(ctx, path, lambda d: unset_in(d, f"providers.{name}"))
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data({"removed": name}, f"removed provider {name}")


@providers.command("login")
@click.argument("name")
@click.option("--from-stdin", is_flag=True, help="Read the key from stdin instead of prompting.")
@click.pass_obj
def providers_login(ctx: CLIContext, name: str, from_stdin: bool) -> None:
    """Store an API key in the local 0600 credential store and reference it as store:NAME."""
    from coremain.security.credentials import CredentialStore

    paths = resolve_core_paths(ctx.env)
    key = (
        sys.stdin.readline().strip()
        if from_stdin
        else getpass.getpass(f"API key for {name} (input hidden): ").strip()
    )
    if not key:
        raise CommandExit(int(ctx.output.error(UsageError("empty key"))))
    CredentialStore(paths.credentials_file).set(name, key)
    cfg = read_toml(paths.config_file)
    if name in (cfg.get("providers") or {}) and not cfg["providers"][name].get("api_key", "").startswith(
        "store:"
    ):
        _validated_update(
            ctx, paths.config_file, lambda d: set_in(d, f"providers.{name}.api_key", f"store:{name}")
        )
    ctx.output.data(
        {"stored": name, "reference": f"store:{name}"},
        f"stored credential for {name} (reference `store:{name}`); the key is never shown again",
    )


@providers.command("logout")
@click.argument("name")
@click.pass_obj
def providers_logout(ctx: CLIContext, name: str) -> None:
    from coremain.security.credentials import CredentialStore

    removed = CredentialStore(resolve_core_paths(ctx.env).credentials_file).delete(name)
    ctx.output.data(
        {"removed": removed},
        f"removed stored credential {name}" if removed else f"no stored credential named {name}",
    )


@providers.command("check")
@click.argument("name", required=False)
@click.option("--live", is_flag=True, help="Make a real request (lists models; no tokens are consumed).")
@runtime_command(require_project=False)
async def providers_check(ctx: CLIContext, rt: Any, name: str | None, live: bool) -> int:
    from coremain.providers.errors import ProviderError

    names = [name] if name else list(rt.config.providers)
    results = []
    worst = 0
    for pid in names:
        if pid not in rt.config.providers:
            raise NotFoundError(f"provider {pid} is not configured")
        entry: dict[str, Any] = {"provider": pid, "credential": rt.registry.credential_status(pid).describe()}
        if live:
            try:
                models = await rt.registry.provider(pid).list_models()
                entry.update(live="ok", models=len(models))
            except ProviderError as exc:
                entry.update(
                    live="error", error_class=exc.error_class.value, message=exc.message, hint=exc.hint
                )
                worst = max(worst, int(exc.exit_code))
        results.append(entry)

    def human(c: Any) -> None:
        for r in results:
            line = f"{r['provider']}: credential {r['credential']}"
            if live:
                line += f"; live {r['live']}" + (
                    f" ({r.get('models')} models)"
                    if r["live"] == "ok"
                    else f" — {r['error_class']}: {r['message']}"
                )
            c.print(line)
            if r.get("hint"):
                c.print(f"  [yellow]hint:[/] {r['hint']}")

    ctx.output.data(results, human)
    return worst


@providers.command("discover")
@click.argument("name")
@click.option("--filter", "pattern", default=None, help="Substring filter on model id.")
@runtime_command(require_project=False)
async def providers_discover(ctx: CLIContext, rt: Any, name: str, pattern: str | None) -> int:
    """List models the provider reports as available (live request)."""
    models = await rt.registry.provider(name).list_models()
    if pattern:
        models = [m for m in models if pattern.lower() in m.id.lower()]
    rows = [m.to_dict() for m in models]
    ctx.output.data(
        rows,
        lambda c: ctx.output.table(
            ["id", "context", "max out", "tools", "vision", "$/Mtok in", "$/Mtok out"],
            [
                [
                    m.id,
                    m.context_window or "",
                    m.max_output_tokens or "",
                    "" if m.supports_tools is None else m.supports_tools,
                    "" if m.supports_vision is None else m.supports_vision,
                    f"{m.input_cost_per_mtok:.3f}" if m.input_cost_per_mtok is not None else "",
                    f"{m.output_cost_per_mtok:.3f}" if m.output_cost_per_mtok is not None else "",
                ]
                for m in models[:300]
            ],
        ),
    )
    return 0


# ============================================================================ models
@click.group()
def models() -> None:
    """Configured models (exact ids), capabilities and routing."""


@models.command("list")
@runtime_command(require_project=False)
async def models_list(ctx: CLIContext, rt: Any) -> int:
    rows = []
    for m in rt.registry.models(include_disabled=True):
        st = rt.stats.get(m.key)
        rows.append(
            {
                **m.to_dict(),
                "calls": st.calls,
                "reliability": round(st.reliability, 3),
                "p50_latency_ms": st.p50_latency_ms,
                "blocked": st.last_blocking_class,
            }
        )
    ctx.output.data(
        rows,
        lambda c: (
            ctx.output.table(
                ["key", "ref", "tier", "context", "tools", "vision", "calls", "reliability", "blocked"],
                [
                    [
                        r["key"],
                        r["ref"],
                        r["tier"],
                        r["context_window"],
                        r["capabilities"]["tool_calling"],
                        r["capabilities"]["vision"],
                        r["calls"],
                        r["reliability"],
                        r["blocked"] or "",
                    ]
                    for r in rows
                ],
            )
            if rows
            else c.print("No models configured. See `core models add --help`.")
        ),
    )
    return 0


@models.command("add")
@click.argument("key")
@click.option("--provider", required=True)
@click.option(
    "--id", "model_id", required=True, help="Exact model identifier at the provider (never guessed)."
)
@click.option("--context", "context_window", type=int, default=None)
@click.option("--max-output", type=int, default=None)
@click.option(
    "--tier",
    type=click.Choice(["frontier", "strong", "standard", "light"]),
    default="standard",
    show_default=True,
)
@click.option(
    "--latency", type=click.Choice(["fast", "standard", "slow"]), default="standard", show_default=True
)
@click.option(
    "--strength", "strengths", multiple=True, help="dimension=score (0-1), e.g. coding=0.9 (repeatable)."
)
@click.option("--cost-in", type=float, default=None, help="USD per million input tokens.")
@click.option("--cost-out", type=float, default=None, help="USD per million output tokens.")
@click.option("--vision/--no-vision", default=False)
@click.option("--tools/--no-tools", default=True)
@click.option("--reasoning", type=click.Choice(["none", "standard", "extended"]), default="standard")
@click.option("--effort", type=click.Choice(["minimal", "low", "medium", "high"]), default=None)
@click.option("--roles", default="", help="Comma-separated role restriction.")
@click.option(
    "--from-discovery", is_flag=True, help="Fill context/cost/capabilities from the provider's live listing."
)
@click.option("--default", "make_default", is_flag=True, help="Also set routing.default_model.")
@click.pass_obj
def models_add(
    ctx: CLIContext,
    key: str,
    provider: str,
    model_id: str,
    context_window: int | None,
    max_output: int | None,
    tier: str,
    latency: str,
    strengths: tuple[str, ...],
    cost_in: float | None,
    cost_out: float | None,
    vision: bool,
    tools: bool,
    reasoning: str,
    effort: str | None,
    roles: str,
    from_discovery: bool,
    make_default: bool,
) -> None:
    discovered = None
    if from_discovery:

        async def discover() -> Any:
            rt = ctx.open_runtime(require_project=False)
            try:
                listing = await rt.registry.provider(provider).list_models()
            finally:
                await rt.close()
            return next((m for m in listing if m.id == model_id), None)

        discovered = run_async(ctx, discover())
        if discovered is None:
            raise CommandExit(
                int(
                    ctx.output.error(
                        NotFoundError(
                            f"provider {provider} does not list a model with id '{model_id}'",
                            hint=f"Run `core providers discover {provider}` to see exact ids.",
                        )
                    )
                )
            )
    entry: dict[str, Any] = {
        "provider": provider,
        "id": model_id,
        "tier": tier,
        "latency": latency,
        "context_window": context_window
        or (discovered.context_window if discovered and discovered.context_window else None),
        "max_output_tokens": max_output
        or (discovered.max_output_tokens if discovered and discovered.max_output_tokens else None),
        "capabilities": {
            "tool_calling": tools if not (discovered and discovered.supports_tools is False) else False,
            "vision": vision or bool(discovered and discovered.supports_vision),
            "reasoning": reasoning,
        },
    }
    if effort:
        entry["capabilities"]["reasoning_effort"] = effort
    if entry["context_window"] is None or entry["max_output_tokens"] is None:
        raise CommandExit(
            int(
                ctx.output.error(
                    UsageError(
                        "--context and --max-output are required (or use --from-discovery when the provider reports them)"
                    )
                )
            )
        )
    cost_in = cost_in if cost_in is not None else (discovered.input_cost_per_mtok if discovered else None)
    cost_out = cost_out if cost_out is not None else (discovered.output_cost_per_mtok if discovered else None)
    if cost_in is not None or cost_out is not None:
        entry["cost"] = {
            k: v for k, v in {"input_per_mtok": cost_in, "output_per_mtok": cost_out}.items() if v is not None
        }
    if strengths:
        entry["strengths"] = {}
        for item in strengths:
            dim, _, score = item.partition("=")
            entry["strengths"][dim.strip()] = float(score)
    if roles:
        entry["roles"] = [r.strip() for r in roles.split(",") if r.strip()]
    path = resolve_core_paths(ctx.env).config_file

    def mutate(d: dict[str, Any]) -> None:
        set_in(d, f"models.{key}", entry)
        if make_default:
            set_in(d, "routing.default_model", key)

    try:
        _validated_update(ctx, path, mutate)
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data({"model": key, "config": entry}, f"added model [bold]{key}[/] → {provider}/{model_id}")


@models.command("remove")
@click.argument("key")
@click.pass_obj
def models_remove(ctx: CLIContext, key: str) -> None:
    path = resolve_core_paths(ctx.env).config_file
    try:
        _validated_update(ctx, path, lambda d: unset_in(d, f"models.{key}"))
    except ConfigError as exc:
        raise CommandExit(int(ctx.output.error(exc))) from exc
    ctx.output.data({"removed": key}, f"removed model {key}")


@models.command("show")
@click.argument("key")
@runtime_command(require_project=False)
async def models_show(ctx: CLIContext, rt: Any, key: str) -> int:
    m = rt.registry.model(key)
    st = rt.stats.get(m.key)
    data = {**m.to_dict(), "stats": st.__dict__ | {"reliability": st.reliability}}
    ctx.output.data(data)
    return 0


@models.command("route")
@click.option(
    "--role",
    default="implementer",
    type=click.Choice(["planner", "implementer", "researcher", "reviewer", "debugger", "summarizer"]),
)
@click.option("--kind", default="feature")
@click.option("--risk", default="normal", type=click.Choice(["low", "normal", "high"]))
@click.option("--context", "context_tokens", type=int, default=0)
@click.option(
    "--independent-of", default=None, help="Model key the choice should be independent from (reviewers)."
)
@runtime_command(require_project=False)
async def models_route(
    ctx: CLIContext, rt: Any, role: str, kind: str, risk: str, context_tokens: int, independent_of: str | None
) -> int:
    """Explain which model the router would choose and why."""
    from coremain.routing.router import RouteRequirements

    decision = rt.router.route(
        RouteRequirements(
            role=role, task_kind=kind, risk=risk, min_context=context_tokens, independent_of=independent_of
        )
    )
    data = decision.to_dict()

    def human(c: Any) -> None:
        c.print(
            f"[bold]{role}[/] → {data['model']} (score {data['score']})"
            + (f" · {data['independence']}" if data.get("independence") else "")
        )
        for r in data["reasons"]:
            c.print(f"  - {r}")
        ctx.output.table(
            ["candidate", "score", "quality", "reliability", "cost", "latency", "rejected"],
            [
                [
                    x["model"],
                    x.get("score", ""),
                    x.get("quality", ""),
                    x.get("reliability", ""),
                    x.get("cost", ""),
                    x.get("latency", ""),
                    x.get("rejected", ""),
                ]
                for x in data["candidates"]
            ],
        )

    ctx.output.data(data, human)
    return 0


@models.command("test")
@click.argument("key")
@runtime_command(require_project=False)
async def models_test(ctx: CLIContext, rt: Any, key: str) -> int:
    """Send one tiny real completion to verify the model end to end (consumes a few tokens)."""
    from coremain.providers.client import CallContext
    from coremain.providers.types import ChatMessage, ChatRequest
    from coremain.runtime.cancel import CancelToken

    m = rt.registry.model(key)
    resp = await rt.client.complete(
        m,
        ChatRequest(
            model=m.model_id, messages=[ChatMessage("user", "Reply with exactly: OK")], max_output_tokens=16
        ),
        CallContext(purpose="models.test", role="summarizer"),
        cancel=CancelToken(),
    )
    data = {
        "model": m.ref,
        "text": resp.text,
        "usage": resp.usage.to_dict(),
        "latency_ms": resp.latency_ms,
        "ttft_ms": resp.ttft_ms,
    }
    ctx.output.data(
        data,
        lambda c: c.print(
            f"{m.ref}: {one_line(resp.text, 80)!r} ({resp.latency_ms} ms, {resp.usage.input_tokens}+{resp.usage.output_tokens} tokens)"
        ),
    )
    return 0


__all__ = ["ago", "config", "models", "permissions", "providers"]
