"""Symbol and import extraction.

Python uses the standard-library ``ast`` (precise). Other languages use conservative regular
expressions (marked ``heuristic`` in parse status) so the index degrades gracefully instead of
pretending to be a full parser. LSP servers, when available, provide precise answers on demand.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str
    line: int
    end_line: int | None = None
    signature: str | None = None
    parent: str | None = None
    exported: bool = True
    doc: str | None = None


@dataclass
class Extraction:
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[tuple[str, int]] = field(default_factory=list)
    status: str = "ok"
    summary: str | None = None


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - unparse can fail on exotic syntax; degrade to an ellipsis
        return "..."


def _py_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    returns = f" -> {_unparse(node.returns)}" if node.returns is not None else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({_unparse(node.args)}){returns}"


def _first_line(doc: str | None) -> str | None:
    return doc.strip().splitlines()[0][:160] if doc and doc.strip() else None


def extract_python(text: str) -> Extraction:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        ex = extract_generic(text, "python")
        ex.status = f"syntax_error: {exc.__class__.__name__}"
        return ex
    result = Extraction()
    result.summary = _first_line(ast.get_docstring(tree))

    def visit(body: list[ast.stmt], parent: str | None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{parent}.{node.name}" if parent else node.name
                decos = ["@" + _unparse(d)[:120] for d in node.decorator_list]
                sig = _py_signature(node)
                if decos:
                    sig = " ".join(decos[:2]) + " " + sig
                result.symbols.append(
                    Symbol(
                        node.name,
                        qual,
                        "method" if parent else "function",
                        node.lineno,
                        getattr(node, "end_lineno", None),
                        sig[:300],
                        parent,
                        not node.name.startswith("_"),
                        _first_line(ast.get_docstring(node)),
                    )
                )
            elif isinstance(node, ast.ClassDef):
                qual = f"{parent}.{node.name}" if parent else node.name
                bases = [_unparse(b) for b in node.bases]
                sig = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
                result.symbols.append(
                    Symbol(
                        node.name,
                        qual,
                        "class",
                        node.lineno,
                        getattr(node, "end_lineno", None),
                        sig,
                        parent,
                        not node.name.startswith("_"),
                        _first_line(ast.get_docstring(node)),
                    )
                )
                visit(node.body, qual)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and parent is None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and (
                        t.id.isupper() or t.id in {"app", "router", "bot", "dp", "client", "api"}
                    ):
                        result.symbols.append(
                            Symbol(
                                t.id,
                                t.id,
                                "constant" if t.id.isupper() else "variable",
                                node.lineno,
                                getattr(node, "end_lineno", None),
                                None,
                                None,
                                not t.id.startswith("_"),
                            )
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                if node.module is None and node.level:
                    for alias in node.names:
                        result.imports.append((base + alias.name, node.lineno))
                else:
                    result.imports.append((base, node.lineno))
            elif isinstance(node, (ast.If, ast.Try)) and parent is None:
                inner: list[ast.stmt] = list(node.body)
                if isinstance(node, ast.Try):
                    for handler in node.handlers:
                        inner.extend(handler.body)
                visit(inner, parent)

    visit(tree.body, None)
    return result


_JS = [
    (
        "function",
        re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(([^)]*)\)",
            re.M,
        ),
    ),
    (
        "class",
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", re.M),
    ),
    (
        "function",
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
            re.M,
        ),
    ),
    ("route", re.compile(r"""\b(?:app|router)\.(get|post|put|patch|delete)\(\s*['"`]([^'"`]+)""", re.M)),
    ("test", re.compile(r"""^\s*(?:it|test|describe)\(\s*['"`]([^'"`]{1,120})""", re.M)),
]
_C = [("function", re.compile(r"^[A-Za-z_][\w\s\*]+?\s+\**([A-Za-z_]\w*)\s*\(([^;{)]*)\)\s*\{", re.M))]
_GENERIC: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "javascript": _JS,
    "typescript": [
        *_JS,
        ("interface", re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)", re.M)),
    ],
    "vue": _JS,
    "svelte": _JS,
    "go": [
        ("function", re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(([^)]*)\)", re.M)),
        ("type", re.compile(r"^type\s+([A-Za-z_]\w*)\s+(struct|interface)", re.M)),
    ],
    "rust": [
        (
            "function",
            re.compile(
                r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*(?:<[^>]*>)?\s*\(([^)]*)\)",
                re.M,
            ),
        ),
        ("type", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)", re.M)),
        ("impl", re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?([A-Za-z_][\w:]*)", re.M)),
    ],
    "java": [
        (
            "class",
            re.compile(
                r"^\s*(?:public|private|protected)?\s*(?:abstract\s+|final\s+|static\s+)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)",
                re.M,
            ),
        ),
        (
            "method",
            re.compile(
                r"^\s*(?:public|private|protected)\s+(?:static\s+|final\s+|synchronized\s+|abstract\s+)*[\w<>\[\],\s]+\s+([a-z_]\w*)\s*\(([^)]*)\)\s*(?:throws[\w\s,]+)?\{",
                re.M,
            ),
        ),
    ],
    "kotlin": [
        (
            "class",
            re.compile(
                r"^\s*(?:data\s+|sealed\s+|open\s+|abstract\s+)*(?:class|interface|object)\s+([A-Za-z_]\w*)",
                re.M,
            ),
        ),
        (
            "function",
            re.compile(r"^\s*(?:suspend\s+)?fun\s+(?:<[^>]*>\s*)?([A-Za-z_]\w*)\s*\(([^)]*)\)", re.M),
        ),
    ],
    "csharp": [
        (
            "class",
            re.compile(
                r"^\s*(?:public|internal|private)?\s*(?:static\s+|abstract\s+|sealed\s+|partial\s+)*(?:class|interface|record|struct)\s+([A-Za-z_]\w*)",
                re.M,
            ),
        ),
        (
            "method",
            re.compile(
                r"^\s*(?:public|private|protected|internal)\s+(?:static\s+|async\s+|virtual\s+|override\s+)*[\w<>\[\],?\s]+\s+([A-Z]\w*)\s*\(([^)]*)\)",
                re.M,
            ),
        ),
    ],
    "ruby": [
        ("class", re.compile(r"^\s*(?:class|module)\s+([A-Z][\w:]*)", re.M)),
        ("method", re.compile(r"^\s*def\s+(?:self\.)?([a-z_]\w*[?!]?)", re.M)),
    ],
    "php": [
        (
            "class",
            re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+([A-Za-z_]\w*)", re.M),
        ),
        (
            "function",
            re.compile(
                r"^\s*(?:public|private|protected)?\s*(?:static\s+)?function\s+([A-Za-z_]\w*)\s*\(([^)]*)\)",
                re.M,
            ),
        ),
    ],
    "c": _C,
    "cpp": _C,
    "shell": [("function", re.compile(r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{", re.M))],
    "sql": [
        (
            "table",
            re.compile(
                r"(?i)^\s*create\s+(?:table|view|index)\s+(?:if\s+not\s+exists\s+)?[`\"]?([\w.]+)", re.M
            ),
        )
    ],
    "markdown": [("heading", re.compile(r"^(#{1,3})\s+(.+)$", re.M))],
}
_JS_IMPORT = re.compile(
    r"""(?:import\s[^'"]*?from\s*|import\s*\(\s*|require\(\s*|export\s[^'"]*?from\s*)['"]([^'"]+)['"]"""
)
_C_IMPORT = re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.M)
_IMPORTS: dict[str, re.Pattern[str]] = {
    "javascript": _JS_IMPORT,
    "typescript": _JS_IMPORT,
    "vue": _JS_IMPORT,
    "svelte": _JS_IMPORT,
    "go": re.compile(r'^\s*(?:import\s+)?(?:[\w.]+\s+)?"([^"]+)"', re.M),
    "rust": re.compile(r"^\s*(?:pub\s+)?(?:use|mod)\s+([\w:]+)", re.M),
    "java": re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)", re.M),
    "kotlin": re.compile(r"^\s*import\s+([\w.]+)", re.M),
    "ruby": re.compile(r"""^\s*require(?:_relative)?\s+['"]([^'"]+)['"]""", re.M),
    "php": re.compile(r"^\s*use\s+([\w\\]+)", re.M),
    "c": _C_IMPORT,
    "cpp": _C_IMPORT,
    "csharp": re.compile(r"^\s*using\s+([\w.]+)\s*;", re.M),
}


def extract_generic(text: str, language: str) -> Extraction:
    result = Extraction(status="heuristic")
    for kind, rx in _GENERIC.get(language, []):
        for m in rx.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            if kind == "heading":
                name = m.group(2).strip()[:120]
                result.symbols.append(Symbol(name, name, "heading", line, signature=m.group(0).strip()[:160]))
                continue
            if kind == "route":
                name = f"{m.group(1).upper()} {m.group(2)}"
                result.symbols.append(Symbol(name, name, "route", line, signature=m.group(0).strip()[:160]))
                continue
            name = m.group(1)
            sig = m.group(0).strip().rstrip("{").strip()[:200]
            exported = not name.startswith("_") and (
                "export" in m.group(0) or language not in {"javascript", "typescript"}
            )
            result.symbols.append(Symbol(name, name, kind, line, signature=sig, exported=exported))
    rx_imp = _IMPORTS.get(language)
    if rx_imp is not None:
        for m in rx_imp.finditer(text):
            result.imports.append((m.group(1), text.count("\n", 0, m.start()) + 1))
    return result


def extract(text: str, language: str | None) -> Extraction:
    if language == "python":
        return extract_python(text)
    if language is None:
        return Extraction(status="unsupported")
    return extract_generic(text, language)
