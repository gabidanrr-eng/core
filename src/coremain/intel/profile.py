"""Project intake: detect ecosystems, frameworks, commands, entry points and conventions.

The profile is compact, cached on the project record and invalidated when any manifest file
changes (its hash covers every manifest considered).
"""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coremain.util.jsonutil import sha256_hex

MANIFESTS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "requirements-dev.txt", "Pipfile", "tox.ini", "pytest.ini",
    "mypy.ini", "ruff.toml", ".flake8", "uv.lock", "poetry.lock", "package.json", "tsconfig.json", "pnpm-lock.yaml", "yarn.lock",
    "package-lock.json", "bun.lockb", "bun.lock", "go.mod", "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts", "Gemfile",
    "composer.json", "Makefile", "Dockerfile", "docker-compose.yml", "compose.yaml", "alembic.ini", "manage.py", "conftest.py",
    "pyrightconfig.json", "deno.json", "vite.config.ts", "vite.config.js", "next.config.js", "playwright.config.ts",
)
FRAMEWORK_HINTS = {
    "fastapi": "FastAPI", "flask": "Flask", "django": "Django", "aiogram": "aiogram (Telegram)", "telegram": "python-telegram-bot",
    "discord": "discord.py", "sqlalchemy": "SQLAlchemy", "alembic": "Alembic", "celery": "Celery", "pydantic": "Pydantic",
    "httpx": "httpx", "requests": "requests", "aiohttp": "aiohttp", "pytest-asyncio": "pytest-asyncio", "typer": "Typer",
    "click": "Click", "textual": "Textual", "scrapy": "Scrapy", "playwright": "Playwright", "selenium": "Selenium",
}
NODE_HINTS = {
    "react": "React", "next": "Next.js", "vue": "Vue", "svelte": "Svelte", "@sveltejs/kit": "SvelteKit", "express": "Express",
    "fastify": "Fastify", "@nestjs/core": "NestJS", "jest": "Jest", "vitest": "Vitest", "@playwright/test": "Playwright",
    "typescript": "TypeScript", "discord.js": "discord.js", "telegraf": "Telegraf", "grammy": "grammY", "prisma": "Prisma",
    "vite": "Vite", "tailwindcss": "Tailwind", "electron": "Electron",
}


@dataclass
class ProjectProfile:
    languages: dict[str, int] = field(default_factory=dict)
    primary_language: str | None = None
    ecosystems: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    commands: dict[str, str] = field(default_factory=dict)
    command_sources: dict[str, str] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    important_files: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    python: dict[str, Any] = field(default_factory=dict)
    ci: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    manifests_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def summary(self) -> str:
        parts = []
        if self.primary_language:
            parts.append(f"Primary language: {self.primary_language}")
        if self.ecosystems:
            parts.append("Ecosystems: " + ", ".join(self.ecosystems))
        if self.frameworks:
            parts.append("Frameworks/libraries: " + ", ".join(self.frameworks[:12]))
        for key in ("test", "lint", "typecheck", "build", "format"):
            if key in self.commands:
                parts.append(f"{key} command: `{self.commands[key]}` ({self.command_sources.get(key, 'detected')})")
        if self.entry_points:
            parts.append("Entry points: " + ", ".join(self.entry_points[:8]))
        if self.test_dirs:
            parts.append("Test locations: " + ", ".join(self.test_dirs[:6]))
        if self.python.get("packages"):
            parts.append("Python packages: " + ", ".join(self.python["packages"][:8]))
        if self.notes:
            parts.append("Notes: " + "; ".join(self.notes[:6]))
        return "\n".join(parts)


def manifests_hash(root: Path) -> str:
    parts = []
    for name in MANIFESTS:
        p = root / name
        if p.is_file():
            try:
                parts.append(f"{name}:{sha256_hex(p.read_bytes())}")
            except OSError:
                continue
    return sha256_hex("|".join(parts))


def _read(root: Path, name: str) -> str:
    try:
        return (root / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _python_runner(root: Path) -> tuple[str, str]:
    if (root / "uv.lock").exists() and shutil.which("uv"):
        return "uv run", "uv"
    if (root / "poetry.lock").exists() and shutil.which("poetry"):
        return "poetry run", "poetry"
    for venv in (".venv", "venv"):
        py = root / venv / "bin" / "python"
        if py.exists():
            return f"{venv}/bin/python -m", "venv"
    return ("python3 -m" if shutil.which("python3") else "python -m"), "system"


def _py_cmd(runner: str, tool: str) -> str:
    if runner.endswith("-m"):
        return f"{runner} {tool}"
    return f"{runner} {tool}"


def detect_profile(root: Path, languages: dict[str, int] | None = None, files: list[str] | None = None) -> ProjectProfile:
    prof = ProjectProfile(languages=dict(languages or {}))
    code_langs = {k: v for k, v in prof.languages.items() if k not in {"markdown", "json", "yaml", "toml", "text", "ini", "html", "css", "rst"}}
    if code_langs:
        prof.primary_language = max(code_langs, key=lambda k: code_langs[k])
    files = files or []
    prof.manifests_hash = manifests_hash(root)
    for name in ("README.md", "README.rst", "README", "AGENTS.md", "CORE.md", "CLAUDE.md", "CONTRIBUTING.md", "Makefile", "Dockerfile",
                 "docker-compose.yml", "pyproject.toml", "package.json", "go.mod", "Cargo.toml", ".env.example"):
        if (root / name).exists():
            prof.important_files.append(name)
    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        prof.ci = sorted(str(p.relative_to(root)) for p in workflows.glob("*.y*ml"))
    test_dirs = sorted({f.split("/")[0] for f in files if re.match(r"^(tests?|spec|__tests__)/", f)})
    prof.test_dirs = test_dirs

    def set_cmd(key: str, value: str, source: str) -> None:
        if key not in prof.commands:
            prof.commands[key] = value
            prof.command_sources[key] = source

    # ---------------------------------------------------------------- python
    pyproject_text = _read(root, "pyproject.toml")
    pyproject: dict[str, Any] = {}
    if pyproject_text:
        try:
            pyproject = tomllib.loads(pyproject_text)
        except tomllib.TOMLDecodeError:
            prof.notes.append("pyproject.toml could not be parsed")
    has_python = bool(pyproject_text) or any((root / n).exists() for n in ("setup.py", "setup.cfg", "requirements.txt", "Pipfile")) or \
        prof.languages.get("python", 0) > 0
    if has_python:
        prof.ecosystems.append("python")
        runner, kind = _python_runner(root)
        prof.package_managers.append(kind if kind != "system" else "pip")
        deps_blob = (pyproject_text + _read(root, "requirements.txt") + _read(root, "requirements-dev.txt") + _read(root, "setup.py")
                     + _read(root, "setup.cfg") + _read(root, "Pipfile")).lower()
        for key, label in FRAMEWORK_HINTS.items():
            if re.search(rf"(?<![\w-]){re.escape(key)}(?![\w-])", deps_blob):
                prof.frameworks.append(label)
        tool = pyproject.get("tool", {}) if isinstance(pyproject.get("tool"), dict) else {}
        uses_pytest = ("pytest" in tool or (root / "pytest.ini").exists() or (root / "conftest.py").exists()
                       or "[pytest]" in _read(root, "tox.ini") or "[tool:pytest]" in _read(root, "setup.cfg") or "pytest" in deps_blob
                       or any(re.search(r"(^|/)test_[^/]+\.py$", f) for f in files))
        if uses_pytest:
            set_cmd("test", _py_cmd(runner, "pytest -q"), "pytest detected")
        elif any(re.search(r"(^|/)tests?/.*\.py$", f) for f in files):
            set_cmd("test", _py_cmd(runner, "unittest discover -s tests" if runner.endswith("-m") else "python -m unittest discover -s tests"),
                    "unittest layout")
        if "ruff" in tool or (root / "ruff.toml").exists() or "ruff" in deps_blob:
            set_cmd("lint", _py_cmd(runner, "ruff check ."), "ruff configured")
            set_cmd("format", _py_cmd(runner, "ruff format --check ."), "ruff configured")
        elif (root / ".flake8").exists() or "flake8" in deps_blob:
            set_cmd("lint", _py_cmd(runner, "flake8"), "flake8 configured")
        if "mypy" in tool or (root / "mypy.ini").exists():
            set_cmd("typecheck", _py_cmd(runner, "mypy ."), "mypy configured")
        elif (root / "pyrightconfig.json").exists() or "pyright" in tool:
            set_cmd("typecheck", _py_cmd(runner, "pyright"), "pyright configured")
        if "black" in tool and "format" not in prof.commands:
            set_cmd("format", _py_cmd(runner, "black --check ."), "black configured")
        packages = sorted({f.split("/")[0] for f in files if f.count("/") == 1 and f.endswith("/__init__.py")}
                          | {f.split("/")[1] for f in files if f.startswith("src/") and f.count("/") == 2 and f.endswith("/__init__.py")})
        src_dirs = ["src"] if any(f.startswith("src/") and f.endswith(".py") for f in files) else []
        prof.python = {"runner": runner, "runner_kind": kind, "packages": packages, "src_dirs": src_dirs,
                       "venv": next((v for v in (".venv", "venv") if (root / v / "bin" / "python").exists()), None)}
        scripts = (pyproject.get("project", {}) or {}).get("scripts", {}) if isinstance(pyproject.get("project"), dict) else {}
        for name, target in list(scripts.items())[:5]:
            prof.entry_points.append(f"console script `{name}` → {target}")
        for cand in ("manage.py", "main.py", "app.py", "bot.py", "server.py", "run.py"):
            if (root / cand).exists():
                prof.entry_points.append(cand)
        for f in files:
            if f.endswith("/__main__.py") and f.count("/") <= 2:
                prof.entry_points.append(f)
        if (root / "alembic.ini").exists():
            prof.notes.append("database migrations managed by Alembic (alembic.ini)")
    # ------------------------------------------------------------------ node
    pkg_text = _read(root, "package.json")
    if pkg_text:
        prof.ecosystems.append("node")
        try:
            pkg = json.loads(pkg_text)
        except json.JSONDecodeError:
            pkg = {}
            prof.notes.append("package.json could not be parsed")
        if (root / "pnpm-lock.yaml").exists():
            pm = "pnpm"
        elif (root / "yarn.lock").exists():
            pm = "yarn"
        elif (root / "bun.lockb").exists() or (root / "bun.lock").exists():
            pm = "bun"
        else:
            pm = "npm"
        prof.package_managers.append(pm)
        scripts = pkg.get("scripts", {}) if isinstance(pkg.get("scripts"), dict) else {}
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        for key, label in NODE_HINTS.items():
            if key in deps:
                prof.frameworks.append(label)
        run = "npm run" if pm == "npm" else f"{pm} run"
        for key, script_names in (("test", ("test",)), ("lint", ("lint",)), ("typecheck", ("typecheck", "type-check", "tsc")),
                                  ("build", ("build",)), ("format", ("format:check", "fmt:check", "prettier:check"))):
            for s in script_names:
                value = scripts.get(s)
                if value and not (key == "test" and "no test specified" in value):
                    set_cmd(key, f"{pm} test" if key == "test" and pm in {"npm", "pnpm", "yarn", "bun"} else f"{run} {s}", f"package.json scripts.{s}")
                    break
        if "typecheck" not in prof.commands and (root / "tsconfig.json").exists():
            set_cmd("typecheck", "npx tsc --noEmit", "tsconfig.json present")
        main = pkg.get("main")
        if isinstance(main, str):
            prof.entry_points.append(main)
        if isinstance(pkg.get("bin"), dict):
            prof.entry_points.extend(f"bin `{k}` → {v}" for k, v in list(pkg["bin"].items())[:4])
    # ------------------------------------------------------------ go / rust
    if (root / "go.mod").exists():
        prof.ecosystems.append("go")
        prof.package_managers.append("go")
        set_cmd("test", "go test ./...", "go.mod")
        set_cmd("build", "go build ./...", "go.mod")
        set_cmd("lint", "go vet ./...", "go.mod")
        prof.entry_points.extend(f for f in files if re.match(r"^(cmd/[^/]+/)?main\.go$", f))
    if (root / "Cargo.toml").exists():
        prof.ecosystems.append("rust")
        prof.package_managers.append("cargo")
        set_cmd("test", "cargo test", "Cargo.toml")
        set_cmd("build", "cargo build", "Cargo.toml")
        set_cmd("lint", "cargo clippy -- -D warnings", "Cargo.toml")
    # ------------------------------------------------------------------ jvm etc.
    if (root / "pom.xml").exists():
        prof.ecosystems.append("java")
        set_cmd("test", "mvn -q test", "pom.xml")
        set_cmd("build", "mvn -q package -DskipTests", "pom.xml")
    elif (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        prof.ecosystems.append("java")
        gradle = "./gradlew" if (root / "gradlew").exists() else "gradle"
        set_cmd("test", f"{gradle} test", "gradle build")
        set_cmd("build", f"{gradle} build -x test", "gradle build")
    if (root / "Gemfile").exists():
        prof.ecosystems.append("ruby")
        set_cmd("test", "bundle exec rspec" if (root / "spec").is_dir() else "bundle exec rake test", "Gemfile")
    if (root / "composer.json").exists():
        prof.ecosystems.append("php")
        set_cmd("test", "vendor/bin/phpunit", "composer.json")
    makefile = _read(root, "Makefile")
    if makefile:
        targets = set(re.findall(r"^([A-Za-z][\w-]*):", makefile, re.M))
        for key, names in (("test", ("test", "check")), ("lint", ("lint",)), ("build", ("build",)), ("typecheck", ("typecheck", "mypy"))):
            for n in names:
                if n in targets:
                    set_cmd(key, f"make {n}", "Makefile target")
                    break
    if (root / "Dockerfile").exists() or (root / "docker-compose.yml").exists() or (root / "compose.yaml").exists():
        prof.notes.append("containerized (Dockerfile/compose present)")
    if (root / ".env.example").exists():
        prof.notes.append("configuration via environment variables (.env.example present; real .env is never read)")
    if not prof.commands.get("test"):
        prof.notes.append("no automated test command detected")
    prof.frameworks = list(dict.fromkeys(prof.frameworks))
    prof.entry_points = list(dict.fromkeys(prof.entry_points))
    return prof
