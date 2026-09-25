"""`core doctor`: inspect every subsystem and say exactly what is wrong and how to fix it.

Checks are read-only. ``live=True`` additionally contacts providers (model listing only, no
tokens consumed), MCP servers and a real headless browser.
"""

from __future__ import annotations

import contextlib
import shutil
import socket
import sqlite3
import stat
import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from coremain.exec.process import process_alive
from coremain.store.migrate import status as migration_status

if TYPE_CHECKING:
    from coremain.runtime.app import CoreRuntime

Check = dict[str, Any]


def _check(name: str, status: str, detail: str, fix: str | None = None) -> Check:
    return {"name": name, "status": status, "detail": detail, **({"fix": fix} if fix else {})}


def _sqlite_features() -> Check:
    conn = sqlite3.connect(":memory:")
    try:
        missing = []
        for probe, name in (("CREATE VIRTUAL TABLE t USING fts5(x)", "FTS5"), ("SELECT json('{}')", "JSON1")):
            try:
                conn.execute(probe)
            except sqlite3.Error:
                missing.append(name)
    finally:
        conn.close()
    if missing:
        return _check(
            "sqlite",
            "fail",
            f"SQLite {sqlite3.sqlite_version} lacks {', '.join(missing)}",
            "use a Python build linked against SQLite ≥ 3.35 with FTS5 and JSON1",
        )
    return _check("sqlite", "ok", f"SQLite {sqlite3.sqlite_version} with FTS5 and JSON1")


def _database(rt: CoreRuntime) -> list[Check]:
    out = []
    problems = rt.db.integrity_check()
    mode = rt.db.scalar("PRAGMA journal_mode")
    out.append(
        _check(
            "database",
            "fail" if problems else "ok",
            "; ".join(problems[:3]) if problems else f"{rt.paths.db_path} (journal {mode})",
            "restore a backup with `core db restore <file>`" if problems else None,
        )
    )
    st = migration_status(rt.db)
    if st.checksum_mismatches:
        out.append(
            _check(
                "migrations",
                "fail",
                f"checksum mismatch for {st.checksum_mismatches}",
                "the database was migrated by a different build; restore a backup or reinstall",
            )
        )
    elif st.pending:
        out.append(
            _check(
                "migrations",
                "warn",
                f"schema v{st.current}, pending {st.pending}",
                "reopen Core Main to migrate",
            )
        )
    else:
        out.append(_check("migrations", "ok", f"schema v{st.current} (latest)"))
    return out


def _config(rt: CoreRuntime) -> list[Check]:
    out = []
    warnings = rt.effective.warnings
    out.append(
        _check(
            "config",
            "warn" if warnings else "ok",
            "; ".join(warnings[:3]) if warnings else "configuration valid",
            "see `core config show --origins`" if warnings else None,
        )
    )
    cred = rt.paths.credentials_file
    if cred.exists():
        mode = stat.S_IMODE(cred.stat().st_mode)
        out.append(
            _check(
                "credential store",
                "ok" if mode & 0o077 == 0 else "fail",
                f"{cred} mode {oct(mode)}",
                None if mode & 0o077 == 0 else f"chmod 600 {cred}",
            )
        )
    if rt.project is not None:
        from coremain.config.loader import project_config_hash
        from coremain.paths import ProjectPaths

        pp = ProjectPaths(rt.project_root) if rt.project_root else None
        has_cfg = bool(pp and (pp.config_file.exists() or pp.local_config_file.exists()))
        if has_cfg and not rt.project_trusted:
            h = project_config_hash(rt.project_root) if rt.project_root else None
            changed = rt.project.trusted_config_hash is not None and rt.project.trusted_config_hash != h
            out.append(
                _check(
                    "project trust",
                    "warn",
                    "project config changed since it was trusted"
                    if changed
                    else "project config is not trusted; "
                    "privileged sections (providers, mcp, permissions, exec, …) are ignored",
                    "core trust",
                )
            )
        else:
            out.append(_check("project trust", "ok", "trusted" if has_cfg else "no project config"))
    offline = rt.config.permissions.network.offline
    out.append(_check("network", "ok", "offline mode: all network access denied" if offline else "online"))
    return out


async def _providers(rt: CoreRuntime, live: bool) -> list[Check]:
    out = []
    if not rt.config.providers:
        return [
            _check(
                "providers",
                "warn",
                "no providers configured",
                "add one with `core providers add` (see README)",
            )
        ]
    for pid, cfg in sorted(rt.config.providers.items()):
        name = f"provider {pid}"
        if not cfg.enabled:
            out.append(_check(name, "skip", "disabled"))
            continue
        cred = rt.registry.credential_status(pid)
        if cfg.api_key and not cred.present:
            out.append(
                _check(
                    name,
                    "fail",
                    f"{cfg.kind} credential {cred.describe()}",
                    f"set the variable or run `core providers login {pid}`",
                )
            )
            continue
        if not live:
            out.append(
                _check(name, "ok", f"{cfg.kind} {cfg.base_url or ''} credential {cred.describe()}".strip())
            )
            continue
        try:
            listed = await rt.registry.provider(pid).list_models()
            out.append(_check(name, "ok", f"reachable; {len(listed)} model(s) listed"))
        except Exception as exc:  # noqa: BLE001 - report the exact provider error class
            cls = getattr(exc, "error_class", None)
            out.append(
                _check(
                    name,
                    "fail",
                    f"{getattr(cls, 'value', cls) or type(exc).__name__}: {exc}",
                    "check base_url, credentials and billing for this provider",
                )
            )
    models = rt.registry.models()
    if not models:
        out.append(
            _check(
                "models",
                "warn",
                "no models configured",
                "add models with exact provider ids (`core models add`)",
            )
        )
    else:
        out.append(
            _check("models", "ok", ", ".join(m.key for m in models[:8]) + (" …" if len(models) > 8 else ""))
        )
        routing = rt.config.routing
        bad = [
            k
            for k in [routing.default_model, *routing.pins.values()]
            if k and k not in {m.key for m in models}
        ]
        if bad:
            out.append(
                _check(
                    "routing",
                    "fail",
                    f"routing refers to unknown/disabled model(s) {bad}",
                    "fix routing.pins/default_model",
                )
            )
    return out


def _tools() -> list[Check]:
    out = []
    git = shutil.which("git")
    out.append(
        _check("git", "ok" if git else "fail", git or "git not found", None if git else "install git ≥ 2.30")
    )
    rg = shutil.which("rg")
    out.append(
        _check(
            "ripgrep",
            "ok" if rg else "warn",
            rg or "not found; text search falls back to Python",
            None if rg else "install ripgrep for faster search",
        )
    )
    out.append(_check("python", "ok" if sys.version_info >= (3, 11) else "fail", sys.version.split()[0]))
    return out


async def _extensions(rt: CoreRuntime, live: bool) -> list[Check]:
    out = []
    if rt.config.mcp:
        manager = rt.extension("mcp")
        for name, cfg in sorted(rt.config.mcp.items()):
            if not cfg.enabled:
                out.append(_check(f"mcp {name}", "skip", "disabled"))
            elif live:
                res = await manager.check(name)
                ok = res["status"] == "ready"
                out.append(
                    _check(
                        f"mcp {name}",
                        "ok" if ok else "fail",
                        f"protocol {res.get('protocol_version')}, {res.get('tools', 0)} tools"
                        if ok
                        else f"{res.get('error_class')}: {res.get('error')}",
                        None if ok else f"core mcp check {name}",
                    )
                )
            else:
                out.append(
                    _check(f"mcp {name}", "ok", f"{cfg.transport}, {cfg.trust} (use --live to connect)")
                )
    for ext, label in (("browser", "browser"), ("lsp", "language servers"), ("research", "docs research")):
        try:
            instance = rt.extension(ext)
        except Exception as exc:  # noqa: BLE001 - an optional subsystem may be unavailable
            out.append(_check(label, "warn", f"unavailable: {exc}"))
            continue
        if ext == "browser":
            res = await instance.check(live=live)
            good = res["status"] in {"available", "ready"}
            out.append(
                _check(
                    label,
                    "ok" if good else ("skip" if res["status"] == "disabled" else "warn"),
                    f"{res['status']}" + (f" — {res.get('detail')}" if res.get("detail") else ""),
                )
            )
        elif ext == "lsp":
            res = instance.status()
            out.append(
                _check(
                    label,
                    "ok" if res.get("available") else "skip",
                    res.get("summary")
                    or res.get("reason")
                    or ("available" if res.get("available") else "none found"),
                )
            )
        else:
            provider = rt.config.research.docs_provider
            out.append(_check(label, "ok" if provider != "none" else "skip", f"docs provider: {provider}"))
    return out


def _skills(rt: CoreRuntime) -> Check:
    skills = rt.skills.all()
    invalid = [s for s in skills if not s.valid]
    untrusted = [s for s in skills if s.valid and not s.usable]
    detail = f"{len(skills)} skill(s); {len(untrusted)} untrusted" + (
        f"; invalid: {', '.join(s.name for s in invalid[:5])}" if invalid else ""
    )
    return _check("skills", "warn" if invalid else "ok", detail, "core skills validate" if invalid else None)


async def _state(rt: CoreRuntime) -> list[Check]:
    out = []
    if rt.project is not None:
        st = rt.index.status(rt.project.id)
        if not st["indexed"]:
            out.append(_check("code index", "warn", "project not indexed yet", "core index build"))
        elif st["indexer_version"] != st["current_version"]:
            out.append(_check("code index", "warn", "built by an older indexer", "core index build --full"))
        else:
            out.append(_check("code index", "ok", f"{st['files']} files, {st['symbols']} symbols"))
    stale = [
        r
        for r in rt.db.query(
            "SELECT id, pid, host FROM runtimes WHERE status = 'running' AND id <> ?", (rt.runtime_id,)
        )
        if r["host"] == socket.gethostname() and not process_alive(int(r["pid"]))
    ]
    live_runtime_ids = {r["id"] for r in rt.db.query("SELECT id FROM runtimes WHERE status = 'running'")}
    orphans = [
        r
        for r in rt.db.query("SELECT pid, runtime_id, command FROM processes WHERE status = 'running'")
        if r["runtime_id"] not in live_runtime_ids and process_alive(int(r["pid"]))
    ]
    out.append(
        _check(
            "runtimes",
            "warn" if stale else "ok",
            f"{len(stale)} dead runtime(s) not yet reconciled" if stale else "no stale runtimes",
            "core recover" if stale else None,
        )
    )
    out.append(
        _check(
            "processes",
            "warn" if orphans else "ok",
            f"{len(orphans)} orphaned process(es): {', '.join(str(o['pid']) for o in orphans[:5])}"
            if orphans
            else "no orphans",
            "core recover --kill-orphans" if orphans else None,
        )
    )
    attention = rt.db.scalar(
        "SELECT COUNT(*) FROM tasks WHERE status IN ('interrupted', 'unknown', 'awaiting_approval', 'needs_input')"
    )
    if attention:
        out.append(_check("tasks", "warn", f"{attention} task(s) need attention", "core status"))
    statuses = {r["id"]: r["status"] for r in rt.db.query("SELECT id, status FROM tasks")}
    actions = await rt.workspaces.gc(task_status=statuses, dry_run=True)
    removable = [a for a in actions if a.action == "delete"]
    out.append(
        _check(
            "workspaces",
            "ok",
            f"{len(removable)} reclaimable workspace(s)" + (" (`core gc --apply`)" if removable else ""),
        )
    )
    problems = rt.artifacts.verify(limit=200)
    out.append(
        _check(
            "artifacts",
            "fail" if problems else "ok",
            "; ".join(problems[:3]) if problems else "sampled blobs verified",
            "core db check --full" if problems else None,
        )
    )
    usage = shutil.disk_usage(rt.paths.data_dir)
    free_gb = usage.free / 1e9
    out.append(
        _check(
            "disk",
            "warn" if free_gb < 1 else "ok",
            f"{free_gb:.1f} GB free at {rt.paths.data_dir}",
            "free disk space or run `core gc --apply`" if free_gb < 1 else None,
        )
    )
    return out


async def run_doctor(rt: CoreRuntime, *, live: bool = False) -> dict[str, Any]:
    checks: list[Check] = [_sqlite_features(), *_database(rt), *_config(rt), *_tools()]
    steps: list[Callable[[], Awaitable[list[Check]]]] = [
        lambda: _providers(rt, live),
        lambda: _extensions(rt, live),
        lambda: _state(rt),
    ]
    for step in steps:
        try:
            checks += await step()
        except Exception as exc:  # noqa: BLE001 - doctor must report, not crash
            checks.append(_check("doctor", "fail", f"check crashed: {type(exc).__name__}: {exc}"))
    with contextlib.suppress(Exception):
        checks.append(_skills(rt))
    failed = sum(1 for c in checks if c["status"] == "fail")
    warnings = sum(1 for c in checks if c["status"] == "warn")
    ok = sum(1 for c in checks if c["status"] == "ok")
    summary = f"{ok} ok, {warnings} warning(s), {failed} failure(s)"
    return {"checks": checks, "failed": failed, "warnings": warnings, "summary": summary}
