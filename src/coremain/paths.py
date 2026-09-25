"""Filesystem locations for Core Main's durable state and project-local configuration.

Durable state (database, artifacts, isolated workspaces, snapshots) lives in the user data
directory, keyed by project identity, so repositories stay clean. ``CORE_HOME`` relocates
everything under a single directory (used by tests, self-test and portable installs).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR_NAME = ".core"


@dataclass(frozen=True)
class CorePaths:
    config_dir: Path
    data_dir: Path
    cache_dir: Path
    state_dir: Path

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def credentials_file(self) -> Path:
        return self.config_dir / "credentials.toml"

    @property
    def user_skills_dir(self) -> Path:
        return self.config_dir / "skills"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "core.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def imported_skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def tmp_dir(self) -> Path:
        return self.cache_dir / "tmp"

    def ensure(self) -> None:
        for d in (
            self.config_dir,
            self.data_dir,
            self.cache_dir,
            self.state_dir,
            self.artifacts_dir,
            self.workspaces_dir,
            self.snapshots_dir,
            self.logs_dir,
            self.tmp_dir,
            self.backups_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        for d in (self.config_dir, self.data_dir):
            try:
                os.chmod(d, 0o700)
            except OSError:
                pass


def resolve_core_paths(env: Mapping[str, str] | None = None) -> CorePaths:
    env = os.environ if env is None else env
    home_override = env.get("CORE_HOME")
    if home_override:
        base = Path(home_override).expanduser().resolve()
        return CorePaths(base / "config", base / "data", base / "cache", base / "state")
    user_home = Path(env.get("HOME") or Path.home())

    def xdg(name: str, default: str) -> Path:
        value = env.get(name)
        return Path(value).expanduser() if value else user_home / default

    return CorePaths(
        config_dir=xdg("XDG_CONFIG_HOME", ".config") / "coremain",
        data_dir=xdg("XDG_DATA_HOME", ".local/share") / "coremain",
        cache_dir=xdg("XDG_CACHE_HOME", ".cache") / "coremain",
        state_dir=xdg("XDG_STATE_HOME", ".local/state") / "coremain",
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def core_dir(self) -> Path:
        return self.root / PROJECT_DIR_NAME

    @property
    def config_file(self) -> Path:
        return self.core_dir / "config.toml"

    @property
    def local_config_file(self) -> Path:
        return self.core_dir / "config.local.toml"

    @property
    def skills_dir(self) -> Path:
        return self.core_dir / "skills"

    @property
    def contracts_dir(self) -> Path:
        return self.core_dir / "contracts"


def find_project_root(start: Path | None = None) -> Path:
    """Nearest ancestor containing ``.core/`` or ``.git``; falls back to ``start`` itself."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / PROJECT_DIR_NAME).is_dir() or (candidate / ".git").exists():
            return candidate
    return here
