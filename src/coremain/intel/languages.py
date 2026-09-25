"""Language detection and file classification."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".vue": "vue",
    ".svelte": "svelte",
    ".lua": "lua",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".tf": "terraform",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".ini": "ini",
    ".cfg": "ini",
    ".txt": "text",
}
NAMED: dict[str, str] = {
    "Dockerfile": "docker",
    "Makefile": "make",
    "Jenkinsfile": "groovy",
    "Gemfile": "ruby",
    "Rakefile": "ruby",
}
CODE_LANGUAGES = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "java",
        "kotlin",
        "ruby",
        "php",
        "csharp",
        "c",
        "cpp",
        "swift",
        "scala",
        "shell",
        "lua",
        "dart",
        "elixir",
        "vue",
        "svelte",
    }
)
_TEST_PATTERNS = [
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)__tests__/"),
    re.compile(r"(^|/)spec/"),
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r"_test\.(py|go)$"),
    re.compile(r"\.(test|spec)\.[cm]?[jt]sx?$"),
    re.compile(r"Test\.(java|kt)$"),
    re.compile(r"_spec\.rb$"),
    re.compile(r"(^|/)conftest\.py$"),
]


def detect_language(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.name in NAMED:
        return NAMED[p.name]
    return EXTENSIONS.get(p.suffix.lower())


def is_test_path(path: str) -> bool:
    return any(rx.search(path) for rx in _TEST_PATTERNS)
