"""Credential references and the local credential store.

Config files hold *references* (``env:NAME``, ``store:NAME``, ``file:PATH``, ``command:CMD``);
secret values are resolved only at the moment of use, registered with the redactor and never
written to the database, events, logs or artifacts.
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from coremain.config.loader import read_toml
from coremain.security.redact import Redactor


@dataclass(frozen=True)
class ResolvedCredential:
    ref: str
    value: str | None
    source: str
    error: str | None = None

    @property
    def present(self) -> bool:
        return bool(self.value)

    def describe(self) -> str:
        if self.present:
            return f"{self.source}: present"
        return f"{self.source}: missing" + (f" ({self.error})" if self.error else "")


class CredentialStore:
    """A 0600 TOML file of named secrets under the user config directory."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict[str, str]:
        data = read_toml(self.path)
        secrets = data.get("secrets", {})
        return {str(k): str(v) for k, v in secrets.items()} if isinstance(secrets, dict) else {}

    def permissions_ok(self) -> bool:
        if not self.path.exists():
            return True
        mode = stat.S_IMODE(self.path.stat().st_mode)
        return mode & 0o077 == 0

    def get(self, name: str) -> str | None:
        return self._load().get(name)

    def names(self) -> list[str]:
        return sorted(self._load())

    def set(self, name: str, value: str) -> None:
        data = self._load()
        data[name] = value
        self._write(data)

    def delete(self, name: str) -> bool:
        data = self._load()
        if name not in data:
            return False
        del data[name]
        self._write(data)
        return True

    def _write(self, secrets: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(tomli_w.dumps({"secrets": secrets}))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)


def resolve_credential(
    ref: str | None,
    store: CredentialStore,
    *,
    env: Mapping[str, str] | None = None,
    redactor: Redactor | None = None,
    command_timeout_s: float = 15.0,
) -> ResolvedCredential:
    if not ref:
        return ResolvedCredential("", None, "none", "no credential configured")
    env = os.environ if env is None else env
    scheme, _, rest = ref.partition(":")
    value: str | None = None
    error: str | None = None
    if scheme == "env":
        value = env.get(rest) or None
        if value is None:
            error = f"environment variable {rest} is not set"
    elif scheme == "store":
        value = store.get(rest)
        if value is None:
            error = f"no stored credential named '{rest}' (use `core providers login`)"
    elif scheme == "file":
        path = Path(rest).expanduser()
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & 0o077:
                error = f"{path} is readable by group/others; chmod 600 it"
            else:
                value = path.read_text(encoding="utf-8").strip() or None
        except OSError as exc:
            error = f"cannot read {path}: {exc.strerror}"
    elif scheme == "command":
        try:
            proc = subprocess.run(
                shlex.split(rest), capture_output=True, text=True, timeout=command_timeout_s, check=False
            )
            if proc.returncode == 0:
                value = proc.stdout.strip() or None
            else:
                error = f"credential command exited with {proc.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"credential command failed: {type(exc).__name__}"
    else:
        error = f"unknown credential scheme '{scheme}'"
    if value and redactor is not None:
        redactor.register_secret(value)
    return ResolvedCredential(ref, value, scheme, error)
