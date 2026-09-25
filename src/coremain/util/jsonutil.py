"""Canonical JSON and hashing helpers."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from pathlib import Path
from typing import Any


def _default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dumps(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(obj, default=_default, ensure_ascii=False, indent=indent)


def canonical(obj: Any) -> str:
    return json.dumps(obj, default=_default, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(text: str | bytes | None, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    return json.loads(text)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    return sha256_hex(canonical(obj))


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
