"""Environment-variable exposure control for subprocesses and external tools."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping

BASE_ALLOW: frozenset[str] = frozenset(
    {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TERM", "TZ", "TMPDIR", "TEMP", "TMP",
        "COLORTERM", "NO_COLOR", "FORCE_COLOR", "CI", "VIRTUAL_ENV", "CONDA_PREFIX", "PYENV_ROOT", "NVM_DIR",
        "GOPATH", "GOROOT", "GOMODCACHE", "GOCACHE", "CARGO_HOME", "RUSTUP_HOME", "JAVA_HOME", "SSL_CERT_FILE",
        "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy", "UV_CACHE_DIR", "PIP_CACHE_DIR", "npm_config_cache",
        "PNPM_HOME", "BUN_INSTALL", "DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "PLAYWRIGHT_BROWSERS_PATH", "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED", "PYTHONIOENCODING", "NODE_ENV", "PAGER", "GIT_PAGER", "EDITOR",
    }
)
BASE_ALLOW_PATTERNS = ("LC_*",)
SECRET_NAME_RE = re.compile(r"(?i)(key|token|secret|passw|pwd|credential|auth|cookie|session|private|signature)")


def is_secret_name(name: str) -> bool:
    return bool(SECRET_NAME_RE.search(name)) and name not in {"PWD", "OLDPWD"}


def build_subprocess_env(
    base: Mapping[str, str],
    *,
    passthrough: Iterable[str] = (),
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Minimal environment: allowlisted variables plus explicit pass-through.

    Secret-looking variables are only passed when listed by exact name in ``passthrough``;
    glob patterns never match secret-looking names.
    """
    explicit = list(passthrough)
    env: dict[str, str] = {}
    for name, value in base.items():
        allowed = name in BASE_ALLOW or any(fnmatch.fnmatchcase(name, p) for p in BASE_ALLOW_PATTERNS)
        if not allowed:
            for pattern in explicit:
                is_glob = any(c in pattern for c in "*?[")
                if pattern == name or (is_glob and fnmatch.fnmatchcase(name, pattern) and not is_secret_name(name)):
                    allowed = True
                    break
        if allowed:
            env[name] = value
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["CORE_MAIN"] = "1"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    if extra:
        env.update(extra)
    return env
