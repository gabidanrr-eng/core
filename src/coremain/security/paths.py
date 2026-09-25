"""Path canonicalization, workspace boundary enforcement and sensitive-path detection."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from coremain.errors import PolicyDeniedError

# Files that commonly hold credentials. Never auto-read, indexed or placed into model context.
SENSITIVE_NAME_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    "_netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".vault-token",
    "*.tfstate",
    "*.tfstate.*",
    "credentials",
    "credentials.json",
    "credentials.toml",
    "*credentials*.json",
    "service-account*.json",
    "*secret*.json",
    "*secrets*.yaml",
    "*secrets*.yml",
    "secrets.toml",
    "auth.json",
    ".htpasswd",
)
SENSITIVE_DIR_NAMES: frozenset[str] = frozenset({".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker", ".gcloud"})
TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist", ".defaults", ".tmpl")


def sensitive_reason(path: str | os.PathLike[str], extra_patterns: Iterable[str] = ()) -> str | None:
    """Return why ``path`` is considered sensitive, or ``None``."""
    p = PurePosixPath(str(path).replace("\\", "/"))
    name = p.name
    for part in p.parts[:-1]:
        if part in SENSITIVE_DIR_NAMES:
            return f"inside credential directory '{part}'"
    lowered = name.lower()
    if lowered.endswith(TEMPLATE_SUFFIXES):
        return None
    for pattern in (*SENSITIVE_NAME_PATTERNS, *extra_patterns):
        if fnmatch.fnmatch(lowered, pattern.lower()) or fnmatch.fnmatch(str(p).lower(), pattern.lower()):
            return f"matches sensitive pattern '{pattern}'"
    return None


class PathViolation(PolicyDeniedError):
    code = "path_violation"


def resolve_within(root: Path, candidate: str | os.PathLike[str]) -> Path:
    """Resolve ``candidate`` relative to ``root`` and refuse anything that escapes it.

    Symlinks are resolved, so a link inside the workspace pointing outside it is rejected.
    """
    raw = os.fspath(candidate)
    if "\x00" in raw:
        raise PathViolation("path contains a NUL byte")
    if not raw.strip():
        raise PathViolation("empty path")
    root_resolved = root.resolve()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = root_resolved / p
    resolved = p.resolve(strict=False)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PathViolation(
            f"path '{raw}' resolves outside the workspace",
            hint="Tools may only access files inside the task workspace.",
            details={"path": raw},
        )
    return resolved


def to_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix() or "."
    except ValueError:
        return path.as_posix()


def is_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
