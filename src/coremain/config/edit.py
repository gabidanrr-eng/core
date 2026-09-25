"""Programmatic, validated edits of TOML config files (used by `core config set`, `core providers add`)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from coremain.config.loader import read_toml
from coremain.errors import ConfigError


def parse_value(raw: str) -> Any:
    """Interpret a CLI value as a TOML literal when possible, else as a plain string."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def set_in(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = [k for k in dotted.split(".") if k]
    if not keys:
        raise ConfigError("empty configuration key")
    cursor = data
    for key in keys[:-1]:
        nxt = cursor.setdefault(key, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"'{key}' in '{dotted}' is not a table")
        cursor = nxt
    cursor[keys[-1]] = value


def unset_in(data: dict[str, Any], dotted: str) -> bool:
    keys = dotted.split(".")
    cursor: Any = data
    for key in keys[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    if isinstance(cursor, dict) and keys[-1] in cursor:
        del cursor[keys[-1]]
        return True
    return False


def write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def update_toml(path: Path, mutate: Any) -> dict[str, Any]:
    data = read_toml(path)
    mutate(data)
    write_toml(path, data)
    return data
