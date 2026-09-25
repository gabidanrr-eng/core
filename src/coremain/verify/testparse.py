"""Best-effort summaries of common test-runner output (pytest, unittest, jest/vitest, go, cargo)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TestSummary:
    runner: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int | None = None
    failures: list[str] | None = None

    @property
    def ran_any(self) -> bool:
        return (self.passed + self.failed + self.errors) > 0

    def line(self) -> str:
        parts = [f"{self.passed} passed"]
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.errors:
            parts.append(f"{self.errors} errors")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        return f"{self.runner}: " + ", ".join(parts)


def _count(pattern: str, text: str) -> int:
    total = 0
    for m in re.finditer(pattern, text):
        total = int(m.group(1))
    return total


def parse_test_output(text: str) -> TestSummary | None:
    if (
        re.search(r"=+ .*(passed|failed|error|no tests ran).* in [\d.]+s", text)
        or "short test summary" in text
        or re.search(r"^\d+ (passed|failed|errors?)\b.* in [\d.]+s", text, re.M)
    ):
        s = TestSummary(
            "pytest",
            _count(r"(\d+) passed", text),
            _count(r"(\d+) failed", text),
            _count(r"(\d+) errors?\b", text),
            _count(r"(\d+) skipped", text),
        )
        s.failures = re.findall(r"^FAILED (\S+)", text, re.M)[:20] or None
        return s
    m = re.search(r"Tests:\s+(?:(\d+) failed, )?(?:(\d+) skipped, )?(?:(\d+) passed, )?(\d+) total", text)
    if m:
        return TestSummary(
            "jest", int(m.group(3) or 0), int(m.group(1) or 0), 0, int(m.group(2) or 0), int(m.group(4))
        )
    m = re.search(r"Test Files .*\n\s+Tests\s+(?:(\d+) failed \| )?(\d+) passed", text)
    if m:
        return TestSummary("vitest", int(m.group(2)), int(m.group(1) or 0))
    if re.search(r"^(ok|FAIL|---) ", text, re.M) and (
        "go test" in text or re.search(r"^ok\s+\S+\s+[\d.]+s", text, re.M)
    ):
        return TestSummary(
            "go",
            len(re.findall(r"^--- PASS", text, re.M)) or len(re.findall(r"^ok\s", text, re.M)),
            len(re.findall(r"^--- FAIL", text, re.M)) or len(re.findall(r"^FAIL\s", text, re.M)),
        )
    m = re.search(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
    if m:
        return TestSummary("cargo", int(m.group(1)), int(m.group(2)), 0, int(m.group(3)))
    m = re.search(
        r"Ran (\d+) tests? in [\d.]+s\s+(OK|FAILED)(?: \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?)?", text
    )
    if m:
        total = int(m.group(1))
        failed = int(m.group(3) or 0)
        errors = int(m.group(4) or 0)
        return TestSummary("unittest", total - failed - errors, failed, errors, 0, total)
    return None
