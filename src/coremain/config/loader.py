"""Configuration layering with per-key provenance.

Precedence (low → high): built-in defaults → user config → project config → project local
config → environment → per-invocation overrides. Project layers come from the repository and
are untrusted until the user runs ``core trust``: they cannot add providers or MCP servers,
cannot relax permissions and cannot point Core Main at new executables.
"""

from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from coremain.config.schema import CoreConfig
from coremain.errors import ConfigError
from coremain.paths import CorePaths, ProjectPaths
from coremain.util.jsonutil import sha256_hex, stable_hash

PROFILE_ORDER = {"read-only": 0, "standard": 1, "autonomous": 2}
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "CORE_MODEL": ("routing", "default_model"),
    "CORE_PERMISSION_PROFILE": ("permissions", "profile"),
    "CORE_WORKSPACE_MODE": ("workspace", "mode"),
    "CORE_OFFLINE": ("permissions", "network", "offline"),
}


@dataclass
class ConfigLayer:
    name: str
    path: Path | None
    data: dict[str, Any]
    trusted: bool = True


@dataclass
class EffectiveConfig:
    config: CoreConfig
    origins: dict[str, str]
    layers: list[ConfigLayer]
    warnings: list[str] = field(default_factory=list)
    project_config_hash: str | None = None
    project_trusted: bool = True

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.config.model_dump(mode="json"))

    def origin_of(self, dotted: str) -> str:
        best = "default"
        for key, origin in self.origins.items():
            if dotted == key or dotted.startswith(key + ".") or key.startswith(dotted + "."):
                best = origin
        return best


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}", hint="Fix the syntax error and retry.") from exc


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            out.update(_flatten(value, dotted))
        else:
            out[dotted] = value
    return out


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key == "rules" and isinstance(value, list) and isinstance(result.get(key), list):
            result[key] = [*result[key], *copy.deepcopy(value)]
        elif isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def restrict_untrusted(data: dict[str, Any], layer: str) -> tuple[dict[str, Any], list[str]]:
    """Drop or narrow settings an untrusted repository must not control."""
    warnings: list[str] = []
    data = copy.deepcopy(data)
    for section in ("providers", "mcp", "exec", "research"):
        if section in data:
            data.pop(section)
            warnings.append(f"{layer}: ignored [{section}] because the project is not trusted (run `core trust`)")
    perms = data.get("permissions")
    if isinstance(perms, dict):
        if "profile" in perms:
            warnings.append(f"{layer}: permissions.profile honoured only if more restrictive (project not trusted)")
            perms["__restrict_profile"] = perms.pop("profile")
        rules = perms.get("rules")
        if isinstance(rules, list):
            kept = [r for r in rules if isinstance(r, dict) and r.get("decision") in {"deny", "ask"}]
            if len(kept) != len(rules):
                warnings.append(f"{layer}: dropped 'allow' permission rules (project not trusted)")
            perms["rules"] = kept
        for key in ("env_passthrough",):
            if key in perms:
                perms.pop(key)
                warnings.append(f"{layer}: ignored permissions.{key} (project not trusted)")
        net = perms.get("network")
        if isinstance(net, dict):
            if "allow_domains" in net:
                net.pop("allow_domains")
                warnings.append(f"{layer}: ignored permissions.network.allow_domains (project not trusted)")
            if net.get("offline") is False:
                net.pop("offline")
    browser = data.get("browser")
    if isinstance(browser, dict):
        for key in ("executable_path", "launch_args", "allowed_origins"):
            if key in browser:
                browser.pop(key)
                warnings.append(f"{layer}: ignored browser.{key} (project not trusted)")
    ws = data.get("workspace")
    if isinstance(ws, dict) and "share_paths" in ws:
        ws.pop("share_paths")
        warnings.append(f"{layer}: ignored workspace.share_paths (project not trusted)")
    intel = data.get("intel")
    if isinstance(intel, dict) and "lsp_servers" in intel:
        intel.pop("lsp_servers")
        warnings.append(f"{layer}: ignored intel.lsp_servers (project not trusted)")
    return data, warnings


def _apply_restricted_profile(merged: dict[str, Any]) -> None:
    perms = merged.get("permissions")
    if not isinstance(perms, dict) or "__restrict_profile" not in perms:
        return
    requested = perms.pop("__restrict_profile")
    current = perms.get("profile", "standard")
    if requested in PROFILE_ORDER and PROFILE_ORDER[requested] < PROFILE_ORDER.get(current, 1):
        perms["profile"] = requested


def _coerce_env(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return value


def project_config_hash(project_root: Path) -> str | None:
    pp = ProjectPaths(project_root)
    parts: list[str] = []
    for path in (pp.config_file, pp.local_config_file):
        if path.exists():
            parts.append(f"{path.name}:{sha256_hex(path.read_bytes())}")
    return sha256_hex("|".join(parts)) if parts else None


def load_config(
    core_paths: CorePaths,
    project_root: Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    project_trusted: bool = False,
) -> EffectiveConfig:
    env = os.environ if env is None else env
    layers: list[ConfigLayer] = [ConfigLayer("user", core_paths.config_file, read_toml(core_paths.config_file))]
    warnings: list[str] = []
    cfg_hash: str | None = None
    if project_root is not None:
        pp = ProjectPaths(project_root)
        cfg_hash = project_config_hash(project_root)
        for name, path in (("project", pp.config_file), ("project-local", pp.local_config_file)):
            data = read_toml(path)
            if not data:
                continue
            if not project_trusted:
                data, w = restrict_untrusted(data, name)
                warnings.extend(w)
            layers.append(ConfigLayer(name, path, data, trusted=project_trusted))
    env_layer: dict[str, Any] = {}
    for var, path_keys in ENV_OVERRIDES.items():
        if env.get(var):
            cursor = env_layer
            for key in path_keys[:-1]:
                cursor = cursor.setdefault(key, {})
            cursor[path_keys[-1]] = _coerce_env(env[var])
    if env_layer:
        layers.append(ConfigLayer("env", None, env_layer))
    if overrides:
        layers.append(ConfigLayer("cli", None, dict(overrides)))

    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for layer in layers:
        merged = deep_merge(merged, layer.data)
        for key in _flatten(layer.data):
            origins[key] = layer.name
    _apply_restricted_profile(merged)
    origins = {k: v for k, v in origins.items() if "__restrict_profile" not in k}
    try:
        config = CoreConfig.model_validate(merged)
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            problems.append(f"{loc or '<root>'}: {err.get('msg')}")
        sources = ", ".join(str(layer.path) for layer in layers if layer.path)
        raise ConfigError(
            "invalid configuration:\n  " + "\n  ".join(problems),
            hint=f"Check {sources or 'your configuration'}; run `core config validate` for details.",
            details={"problems": problems},
        ) from exc
    return EffectiveConfig(config, origins, layers, warnings, cfg_hash, project_trusted)
