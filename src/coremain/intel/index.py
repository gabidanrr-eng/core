"""Incremental code index stored in SQLite.

Change detection uses (size, mtime) first and content hash second, so re-indexing an
unchanged repository is a stat pass. Sensitive files are never indexed. Content is chunked
into FTS5 rows (body, expanded identifiers, path) for lexical retrieval; symbols and imports
feed outlines, the import graph and change-impact analysis.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import posixpath
import subprocess
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coremain.config.schema import IntelConfig
from coremain.intel.extract import extract
from coremain.intel.languages import CODE_LANGUAGES, detect_language, is_test_path
from coremain.security.paths import sensitive_reason
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.jsonutil import sha256_hex
from coremain.util.text import fts_query, identifier_terms, query_terms

INDEXER_VERSION = "3"
CHUNK_LINES = 60
CHUNK_OVERLAP = 12
IGNORED_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".cache",
        ".gradle",
        ".idea",
        "coverage",
        ".turbo",
        "vendor",
        ".core",
    }
)
GENERATED_PATTERNS = (
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "*.pb.go",
    "*_pb2.py",
    "*.snap",
)


@dataclass
class IndexReport:
    scanned: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: int = 0
    duration_ms: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SearchHit:
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str


def _python_module(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    mod = path[:-3]
    for prefix in ("src/", "lib/"):
        if mod.startswith(prefix):
            mod = mod[len(prefix) :]
            break
    if mod.endswith("/__init__"):
        mod = mod[: -len("/__init__")]
    return mod.replace("/", ".")


class CodeIndex:
    def __init__(self, db: Database, config: IntelConfig, clock: Clock):
        self.db = db
        self.config = config
        self.clock = clock

    # ------------------------------------------------------------------ scanning
    def scan_files(self, root: Path) -> list[str]:
        files: list[str] = []
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
                capture_output=True,
                timeout=60,
                check=False,
            )
            if proc.returncode == 0:
                files = [p for p in proc.stdout.decode("utf-8", errors="replace").split("\x00") if p]
        except (OSError, subprocess.TimeoutExpired):
            files = []
        if not files:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
                rel_dir = os.path.relpath(dirpath, root)
                for name in filenames:
                    files.append(name if rel_dir == "." else f"{rel_dir}/{name}".replace(os.sep, "/"))
        out = []
        for rel in files:
            parts = rel.split("/")
            if any(p in IGNORED_DIRS for p in parts[:-1]):
                continue
            if sensitive_reason(rel) or any(fnmatch.fnmatch(parts[-1], g) for g in GENERATED_PATTERNS):
                continue
            if any(fnmatch.fnmatch(rel, pat) for pat in self.config.exclude):
                continue
            out.append(rel)
            if len(out) >= self.config.max_files:
                break
        return sorted(out)

    # ------------------------------------------------------------------ updating
    def update(self, project_id: str, root: Path, *, full: bool = False) -> IndexReport:
        started = time.monotonic()
        report = IndexReport()
        root = root.resolve()
        state = self.db.one("SELECT indexer_version FROM index_state WHERE project_id = ?", (project_id,))
        if state is None or state["indexer_version"] != INDEXER_VERSION:
            full = True
        if full:
            self.clear(project_id)
        existing = {
            r["path"]: r
            for r in self.db.query(
                "SELECT path, sha256, size, mtime_ns FROM index_files WHERE project_id = ?", (project_id,)
            )
        }
        files = self.scan_files(root)
        report.scanned = len(files)
        seen = set(files)
        changed: list[tuple[str, bytes, os.stat_result]] = []
        for rel in files:
            path = root / rel
            try:
                st = path.stat()
            except OSError:
                continue
            if not path.is_file() or st.st_size > self.config.max_file_bytes:
                report.skipped += 1
                continue
            prev = existing.get(rel)
            if prev is not None and prev["size"] == st.st_size and prev["mtime_ns"] == st.st_mtime_ns:
                report.unchanged += 1
                continue
            try:
                data = path.read_bytes()
            except OSError as exc:
                report.errors.append(f"{rel}: {exc.strerror}")
                continue
            if b"\x00" in data[:4096]:
                report.skipped += 1
                continue
            if prev is not None and prev["sha256"] == sha256_hex(data):
                self.db.execute(
                    "UPDATE index_files SET mtime_ns = ? WHERE project_id = ? AND path = ?",
                    (st.st_mtime_ns, project_id, rel),
                )
                report.unchanged += 1
                continue
            changed.append((rel, data, st))
        removed = [p for p in existing if p not in seen]
        report.removed = len(removed)
        for batch_start in range(0, len(removed), 200):
            with self.db.tx() as conn:
                for rel in removed[batch_start : batch_start + 200]:
                    self._delete_file(conn, project_id, rel)
        now = self.clock.now()
        for batch_start in range(0, len(changed), 100):
            with self.db.tx() as conn:
                for rel, data, st in changed[batch_start : batch_start + 100]:
                    self._delete_file(conn, project_id, rel)
                    self._index_file(conn, project_id, rel, data, st, now)
                    report.changed += 1
        if changed or removed:
            self._resolve_imports(project_id, root)
        head = None
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            head = proc.stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            head = None
        report.duration_ms = int((time.monotonic() - started) * 1000)
        counts = self.db.one(
            "SELECT COUNT(*) AS files, COALESCE(SUM(symbol_count), 0) AS symbols FROM index_files WHERE project_id = ?",
            (project_id,),
        )
        self.db.execute(
            "INSERT INTO index_state(project_id, indexer_version, files, symbols, head_ref, last_scan_at, last_duration_ms) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(project_id) DO UPDATE SET indexer_version = excluded.indexer_version, files = excluded.files, "
            "symbols = excluded.symbols, head_ref = excluded.head_ref, last_scan_at = excluded.last_scan_at, "
            "last_duration_ms = excluded.last_duration_ms",
            (
                project_id,
                INDEXER_VERSION,
                counts["files"] if counts else 0,
                counts["symbols"] if counts else 0,
                head,
                now,
                report.duration_ms,
            ),
        )
        return report

    async def update_async(self, project_id: str, root: Path, *, full: bool = False) -> IndexReport:
        return await asyncio.to_thread(self.update, project_id, root, full=full)

    def _delete_file(self, conn: Any, project_id: str, rel: str) -> None:
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM index_chunks WHERE project_id = ? AND path = ?", (project_id, rel)
            ).fetchall()
        ]
        if ids:
            conn.executemany("DELETE FROM index_fts WHERE rowid = ?", [(i,) for i in ids])
        conn.execute("DELETE FROM index_chunks WHERE project_id = ? AND path = ?", (project_id, rel))
        conn.execute("DELETE FROM index_symbols WHERE project_id = ? AND path = ?", (project_id, rel))
        conn.execute("DELETE FROM index_imports WHERE project_id = ? AND path = ?", (project_id, rel))
        conn.execute("DELETE FROM index_files WHERE project_id = ? AND path = ?", (project_id, rel))

    def _index_file(
        self, conn: Any, project_id: str, rel: str, data: bytes, st: os.stat_result, now: float
    ) -> None:
        text = data.decode("utf-8", errors="replace")
        language = detect_language(rel)
        ex = (
            extract(text, language) if language in CODE_LANGUAGES or language in {"sql", "markdown"} else None
        )
        symbols = ex.symbols if ex else []
        conn.execute(
            "INSERT INTO index_files(project_id, path, sha256, size, mtime_ns, language, is_test, parse_status, symbol_count, summary, indexed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                project_id,
                rel,
                sha256_hex(data),
                st.st_size,
                st.st_mtime_ns,
                language,
                1 if is_test_path(rel) else 0,
                ex.status if ex else "text",
                len(symbols),
                ex.summary if ex else None,
                now,
            ),
        )
        if symbols:
            conn.executemany(
                "INSERT INTO index_symbols(project_id, path, name, qualname, kind, line, end_line, signature, parent, exported, doc) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        project_id,
                        rel,
                        s.name,
                        s.qualname,
                        s.kind,
                        s.line,
                        s.end_line,
                        s.signature,
                        s.parent,
                        1 if s.exported else 0,
                        s.doc,
                    )
                    for s in symbols
                ],
            )
        if ex and ex.imports:
            conn.executemany(
                "INSERT INTO index_imports(project_id, path, module, resolved_path, line) VALUES (?,?,?,?,?)",
                [(project_id, rel, mod, None, line) for mod, line in ex.imports],
            )
        lines = text.splitlines()
        step = CHUNK_LINES - CHUNK_OVERLAP
        for start in range(0, max(1, len(lines)), step):
            chunk = lines[start : start + CHUNK_LINES]
            if not chunk and start > 0:
                break
            body = "\n".join(chunk)
            cur = conn.execute(
                "INSERT INTO index_chunks(project_id, path, start_line, end_line) VALUES (?,?,?,?)",
                (project_id, rel, start + 1, start + len(chunk)),
            )
            conn.execute(
                "INSERT INTO index_fts(rowid, body, idents, path) VALUES (?,?,?,?)",
                (cur.lastrowid, body, identifier_terms(body, limit=400), rel.replace("/", " ")),
            )
            if start + CHUNK_LINES >= len(lines):
                break

    def _resolve_imports(self, project_id: str, root: Path) -> None:
        files = {
            r["path"]: r["language"]
            for r in self.db.query(
                "SELECT path, language FROM index_files WHERE project_id = ?", (project_id,)
            )
        }
        py_modules = {m: p for p in files if (m := _python_module(p))}
        go_module = None
        go_mod = root / "go.mod"
        if go_mod.exists():
            for line in go_mod.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("module "):
                    go_module = line.split()[1].strip()
                    break
        rows = self.db.query(
            "SELECT rowid, path, module FROM index_imports WHERE project_id = ?", (project_id,)
        )
        updates: list[tuple[str | None, int]] = []
        for r in rows:
            updates.append(
                (self._resolve_one(r["path"], r["module"], files, py_modules, go_module), r["rowid"])
            )
        with self.db.tx() as conn:
            conn.executemany("UPDATE index_imports SET resolved_path = ? WHERE rowid = ?", updates)

    @staticmethod
    def _resolve_one(
        src: str, module: str, files: dict[str, str | None], py_modules: dict[str, str], go_module: str | None
    ) -> str | None:
        lang = files.get(src)
        if lang == "python":
            if module.startswith("."):
                level = len(module) - len(module.lstrip("."))
                pkg = (_python_module(src) or "").split(".")
                if not src.endswith("__init__.py"):
                    pkg = pkg[:-1]
                base = pkg[: len(pkg) - (level - 1)] if level > 1 else pkg
                rest = module.lstrip(".")
                module = ".".join([*base, *([rest] if rest else [])])
            parts = module.split(".")
            while parts:
                candidate = ".".join(parts)
                if candidate in py_modules:
                    return py_modules[candidate]
                parts.pop()
            return None
        if lang in {"javascript", "typescript", "vue", "svelte"} and module.startswith("."):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(src), module))
            for cand in (
                target,
                *(target + ext for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".vue", ".svelte")),
                *(f"{target}/index{ext}" for ext in (".ts", ".tsx", ".js", ".jsx")),
            ):
                if cand in files:
                    return cand
            return None
        if lang == "go" and go_module and module.startswith(go_module):
            rel_dir = module[len(go_module) :].lstrip("/")
            return next(
                (
                    p
                    for p in sorted(files)
                    if posixpath.dirname(p) == rel_dir and p.endswith(".go") and not p.endswith("_test.go")
                ),
                None,
            )
        if lang in {"ruby"} and module.startswith("."):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(src), module))
            return target + ".rb" if target + ".rb" in files else None
        return None

    # ------------------------------------------------------------------ queries
    def clear(self, project_id: str) -> None:
        with self.db.tx() as conn:
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM index_chunks WHERE project_id = ?", (project_id,)
                ).fetchall()
            ]
            for start in range(0, len(ids), 500):
                conn.executemany(
                    "DELETE FROM index_fts WHERE rowid = ?", [(i,) for i in ids[start : start + 500]]
                )
            for table in ("index_chunks", "index_symbols", "index_imports", "index_files", "index_state"):
                conn.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))

    def status(self, project_id: str) -> dict[str, Any]:
        state = self.db.one("SELECT * FROM index_state WHERE project_id = ?", (project_id,))
        langs = self.languages(project_id)
        parse = {
            r["parse_status"]: r["n"]
            for r in self.db.query(
                "SELECT parse_status, COUNT(*) AS n FROM index_files WHERE project_id = ? GROUP BY parse_status",
                (project_id,),
            )
        }
        return {
            "indexed": state is not None,
            "indexer_version": state["indexer_version"] if state else None,
            "current_version": INDEXER_VERSION,
            "files": state["files"] if state else 0,
            "symbols": state["symbols"] if state else 0,
            "last_scan_at": state["last_scan_at"] if state else None,
            "last_duration_ms": state["last_duration_ms"] if state else None,
            "head_ref": state["head_ref"] if state else None,
            "languages": langs,
            "parse_status": parse,
        }

    def languages(self, project_id: str) -> dict[str, int]:
        return {
            r["language"]: r["n"]
            for r in self.db.query(
                "SELECT language, COUNT(*) AS n FROM index_files WHERE project_id = ? AND language IS NOT NULL GROUP BY language ORDER BY n DESC",
                (project_id,),
            )
        }

    def files(self, project_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.query(
                "SELECT path, language, is_test, symbol_count, size FROM index_files WHERE project_id = ? ORDER BY path",
                (project_id,),
            )
        ]

    def search(self, project_id: str, query: str, *, limit: int = 20) -> list[SearchHit]:
        match = fts_query(query_terms(query))
        if not match:
            return []
        rows = self.db.query(
            "SELECT c.path, c.start_line, c.end_line, bm25(index_fts, 1.0, 1.6, 4.0) AS score, "
            "snippet(index_fts, 0, '', '', ' … ', 24) AS snip FROM index_fts JOIN index_chunks c ON c.id = index_fts.rowid "
            "WHERE index_fts MATCH ? AND c.project_id = ? ORDER BY score LIMIT ?",
            (match, project_id, limit * 3),
        )
        seen: Counter[str] = Counter()
        hits: list[SearchHit] = []
        for r in rows:
            if seen[r["path"]] >= 2:
                continue
            seen[r["path"]] += 1
            hits.append(SearchHit(r["path"], r["start_line"], r["end_line"], -float(r["score"]), r["snip"]))
            if len(hits) >= limit:
                break
        return hits

    def find_symbols(
        self, project_id: str, name: str, *, exact: bool = False, limit: int = 50
    ) -> list[dict[str, Any]]:
        if exact:
            rows = self.db.query(
                "SELECT * FROM index_symbols WHERE project_id = ? AND (name = ? OR qualname = ?) ORDER BY exported DESC, path LIMIT ?",
                (project_id, name, name, limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM index_symbols WHERE project_id = ? AND (name LIKE ? OR qualname LIKE ?) "
                "ORDER BY (name = ?) DESC, exported DESC, length(name), path LIMIT ?",
                (project_id, f"%{name}%", f"%{name}%", name, limit),
            )
        return [
            {k: r[k] for k in ("path", "name", "qualname", "kind", "line", "end_line", "signature", "doc")}
            for r in rows
        ]

    def outline(self, project_id: str, path: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT name, qualname, kind, line, end_line, signature, doc FROM index_symbols WHERE project_id = ? AND path = ? ORDER BY line",
            (project_id, path),
        )
        return [dict(r) for r in rows]

    def outline_text(self, project_id: str, path: str, *, max_items: int = 80) -> str:
        items = self.outline(project_id, path)
        if not items:
            return ""
        lines = []
        for item in items[:max_items]:
            indent = "    " * item["qualname"].count(".")
            sig = item["signature"] or f"{item['kind']} {item['name']}"
            doc = f"  # {item['doc']}" if item.get("doc") else ""
            lines.append(f"{indent}L{item['line']}: {sig}{doc}")
        if len(items) > max_items:
            lines.append(f"… {len(items) - max_items} more symbols")
        return "\n".join(lines)

    def imports_of(self, project_id: str, path: str) -> list[str]:
        return sorted(
            {
                r["resolved_path"]
                for r in self.db.query(
                    "SELECT resolved_path FROM index_imports WHERE project_id = ? AND path = ? AND resolved_path IS NOT NULL",
                    (project_id, path),
                )
            }
        )

    def importers_of(self, project_id: str, path: str) -> list[str]:
        return sorted(
            {
                r["path"]
                for r in self.db.query(
                    "SELECT path FROM index_imports WHERE project_id = ? AND resolved_path = ?",
                    (project_id, path),
                )
            }
        )

    def related_tests(self, project_id: str, path: str) -> list[str]:
        tests = {
            r["path"]
            for r in self.db.query(
                "SELECT i.path FROM index_imports i JOIN index_files f ON f.project_id = i.project_id AND f.path = i.path "
                "WHERE i.project_id = ? AND i.resolved_path = ? AND f.is_test = 1",
                (project_id, path),
            )
        }
        stem = posixpath.splitext(posixpath.basename(path))[0]
        for r in self.db.query(
            "SELECT path FROM index_files WHERE project_id = ? AND is_test = 1 AND (path LIKE ? OR path LIKE ? OR path LIKE ?)",
            (project_id, f"%test_{stem}%", f"%{stem}_test%", f"%{stem}.test.%"),
        ):
            tests.add(r["path"])
        tests.discard(path)
        return sorted(tests)

    def impact(self, project_id: str, paths: list[str], *, depth: int = 3) -> dict[str, Any]:
        reverse: dict[str, set[str]] = defaultdict(set)
        for r in self.db.query(
            "SELECT path, resolved_path FROM index_imports WHERE project_id = ? AND resolved_path IS NOT NULL",
            (project_id,),
        ):
            reverse[r["resolved_path"]].add(r["path"])
        affected: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((p, 0) for p in paths)
        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            for importer in reverse.get(current, ()):
                if importer not in affected and importer not in paths:
                    affected[importer] = d + 1
                    queue.append((importer, d + 1))
        tests: set[str] = set()
        for p in [*paths, *affected]:
            tests.update(self.related_tests(project_id, p))
        test_flags = {
            r["path"]
            for r in self.db.query(
                "SELECT path FROM index_files WHERE project_id = ? AND is_test = 1", (project_id,)
            )
        }
        tests.update(p for p in [*paths, *affected] if p in test_flags)
        return {
            "changed": paths,
            "affected": [
                {"path": p, "distance": d} for p, d in sorted(affected.items(), key=lambda kv: (kv[1], kv[0]))
            ],
            "tests": sorted(tests),
        }

    def centrality(self, project_id: str) -> dict[str, int]:
        return {
            r["resolved_path"]: r["n"]
            for r in self.db.query(
                "SELECT resolved_path, COUNT(DISTINCT path) AS n FROM index_imports WHERE project_id = ? AND resolved_path IS NOT NULL "
                "GROUP BY resolved_path",
                (project_id,),
            )
        }

    def repo_map(self, project_id: str, *, focus: list[str] | None = None, max_files: int = 40) -> str:
        rows = self.db.query(
            "SELECT path, language, is_test, symbol_count, summary FROM index_files WHERE project_id = ?",
            (project_id,),
        )
        if not rows:
            return "(index empty)"
        central = self.centrality(project_id)
        focus_dirs = {posixpath.dirname(f) for f in (focus or [])}
        dir_counts: Counter[str] = Counter(posixpath.dirname(r["path"]) or "." for r in rows)
        top_dirs: Counter[str] = Counter((r["path"].split("/")[0] if "/" in r["path"] else ".") for r in rows)
        lines = [
            f"{len(rows)} indexed files. Top-level: "
            + ", ".join(f"{d}/ ({n})" if d != "." else f"root ({n})" for d, n in top_dirs.most_common(14))
        ]

        def importance(r: Any) -> float:
            p = r["path"]
            score = central.get(p, 0) * 2.0 + min(r["symbol_count"], 30) * 0.1
            if posixpath.dirname(p) in focus_dirs:
                score += 3.0
            if r["is_test"]:
                score -= 1.0
            if posixpath.basename(p) in {
                "main.py",
                "app.py",
                "__main__.py",
                "cli.py",
                "index.ts",
                "main.go",
                "main.rs",
                "server.py",
                "manage.py",
            }:
                score += 2.0
            return score

        ranked = sorted((r for r in rows if r["language"] in CODE_LANGUAGES), key=importance, reverse=True)[
            :max_files
        ]
        for r in sorted(ranked, key=lambda r: r["path"]):
            syms = self.db.query(
                "SELECT kind, name FROM index_symbols WHERE project_id = ? AND path = ? AND exported = 1 AND parent IS NULL "
                "AND kind IN ('class','function','type','interface','route','method') ORDER BY line LIMIT 6",
                (project_id, r["path"]),
            )
            names = ", ".join(f"{s['kind'][:5]} {s['name']}" for s in syms)
            used = f" [imported by {central[r['path']]}]" if central.get(r["path"]) else ""
            summary = f" — {r['summary']}" if r["summary"] else ""
            lines.append(f"{r['path']}{used}{summary}" + (f": {names}" if names else ""))
        busy = ", ".join(f"{d or '.'} ({n})" for d, n in dir_counts.most_common(8))
        lines.append(f"Busiest directories: {busy}")
        return "\n".join(lines)

    def file_info(self, project_id: str, path: str) -> dict[str, Any] | None:
        r = self.db.one("SELECT * FROM index_files WHERE project_id = ? AND path = ?", (project_id, path))
        return dict(r) if r else None
