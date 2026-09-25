"""Deterministic review checks over the actual diff and the implementer's claims."""

from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex
from dataclasses import dataclass
from typing import Any

from coremain.intel.languages import is_test_path
from coremain.security.redact import Redactor
from coremain.workspaces.manager import WorkspaceDiff

SKIP_MARKERS = re.compile(r"(@pytest\.mark\.skip|pytest\.skip\(|@unittest\.skip|\bxit\(|\bxdescribe\(|\b(it|test|describe)\.skip\(|"
                          r"t\.Skip\(|#\[ignore\]|@Disabled)")
DEBUG_LEFTOVERS = re.compile(r"(\bbreakpoint\(\)|pdb\.set_trace\(|debugger;|console\.log\(|dbg!\()")
LOCKFILES = {"package-lock.json": "package.json", "pnpm-lock.yaml": "package.json", "yarn.lock": "package.json",
             "poetry.lock": "pyproject.toml", "uv.lock": "pyproject.toml", "Cargo.lock": "Cargo.toml", "go.sum": "go.mod"}


@dataclass
class Finding:
    severity: str
    category: str
    title: str
    rationale: str
    file: str | None = None
    line: int | None = None
    evidence: str = ""
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _added_lines(patch: str) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    current = None
    line_no = 0
    for raw in patch.splitlines():
        if raw.startswith("+++ b/"):
            current = raw[6:]
        elif raw.startswith("+++ /dev/null"):
            current = None
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            line_no = int(m.group(1)) - 1 if m else 0
        elif current is not None:
            if raw.startswith("+") and not raw.startswith("+++"):
                line_no += 1
                out.append((current, line_no, raw[1:]))
            elif not raw.startswith("-"):
                line_no += 1
    return out


def _norm_cmd(cmd: str) -> str:
    try:
        return " ".join(shlex.split(cmd))
    except ValueError:
        return " ".join(cmd.split())


def static_review(diff: WorkspaceDiff, *, result: dict[str, Any] | None, commands_run: list[dict[str, Any]], contract: dict[str, Any],
                  redactor: Redactor) -> list[Finding]:
    findings: list[Finding] = []
    added = _added_lines(diff.patch)
    changed = {f.path: f.status for f in diff.files}
    for path, line, text in added:
        hits = redactor.scan(text)
        if hits:
            findings.append(Finding("critical", "security", "Possible secret committed", f"Added line matches {hits[0].kind} pattern.",
                                    path, line, "[redacted]", "Remove the secret, load it from the environment or a secret store, and rotate it."))
    for pattern in contract.get("forbidden_paths") or []:
        for path in changed:
            if fnmatch.fnmatch(path, pattern):
                findings.append(Finding("critical", "requirements", f"Forbidden path modified: {path}",
                                        f"The task contract forbids changes matching '{pattern}'.", path, None, f"{changed[path]} {path}",
                                        "Revert changes to this path."))
    for path, status in changed.items():
        if status == "D" and is_test_path(path) and not path.endswith("conftest.py"):
            findings.append(Finding("high", "tests", f"Test file deleted: {path}", "Deleting tests can hide regressions.", path, None,
                                    f"D {path}", "Restore the test or justify its removal explicitly."))
    for path, line, text in added:
        if SKIP_MARKERS.search(text):
            findings.append(Finding("high", "tests", "Test disabled/skipped", "A skip marker was added, which can mask failures.", path, line,
                                    text.strip()[:160], "Remove the skip and fix the underlying failure, or document why it is necessary."))
        if DEBUG_LEFTOVERS.search(text) and not is_test_path(path):
            findings.append(Finding("low", "maintainability", "Debug statement left in code", "Debug statements should not ship.", path, line,
                                    text.strip()[:160], "Remove the debug statement or replace it with proper logging."))
        if re.search(r"\b(TODO|FIXME|XXX)\b", text):
            findings.append(Finding("info", "maintainability", "New TODO/FIXME", "Unfinished work was left in the code.", path, line,
                                    text.strip()[:160], "Resolve it or record it as remaining work."))
    for lock, manifest in LOCKFILES.items():
        for path in changed:
            if posixpath.basename(path) == lock and posixpath.join(posixpath.dirname(path), manifest).lstrip("/") not in changed:
                findings.append(Finding("medium", "dependency", f"Lockfile changed without manifest change: {path}",
                                        "Dependency resolution changed without an explicit dependency change.", path, None, "",
                                        "Confirm the lockfile change is intended; otherwise restore it."))
    if result:
        claimed_files = {str(p).lstrip("./") for p in result.get("files_changed") or []}
        missing = sorted(p for p in claimed_files if p not in changed)
        if missing:
            findings.append(Finding("medium", "misleading_claim", "Claimed changes not present in the diff",
                                    "The implementer reported changing files that are unchanged relative to the task baseline.", None, None,
                                    ", ".join(missing[:10]), "Correct the summary or make the missing changes."))
        claimed_tests = [str(t) for t in result.get("tests_added_or_changed") or []]
        test_changes = [p for p in changed if is_test_path(p)]
        if claimed_tests and not test_changes:
            findings.append(Finding("high", "misleading_claim", "Claimed test changes but no test files changed",
                                    "The result claims tests were added or changed, but no test file appears in the diff.", None, None,
                                    ", ".join(claimed_tests[:5]), "Add the tests or remove the claim."))
        ran = {_norm_cmd(c.get("command", "")) for c in commands_run}
        unverified = [c for c in result.get("commands_run") or [] if _norm_cmd(str(c)) not in ran
                      and not any(_norm_cmd(str(c)) in r or r in _norm_cmd(str(c)) for r in ran if r)]
        if unverified:
            findings.append(Finding("medium", "misleading_claim", "Claimed commands were not executed",
                                    "These commands do not appear in the runtime's command records for this attempt.", None, None,
                                    "; ".join(str(c) for c in unverified[:5]), "Run the commands or remove the claim."))
    big = [f.path for f in diff.files if f.status in {"A", "M"}]
    if diff.stats.get("added", 0) > 3000:
        findings.append(Finding("info", "maintainability", "Very large change", f"{diff.stats.get('added')} lines added.", None, None,
                                ", ".join(big[:5]), "Consider splitting the change."))
    return findings
