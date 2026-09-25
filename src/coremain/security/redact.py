"""Secret detection and redaction for logs, events, artifacts, tool output and model input."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    start: int
    end: int
    line: int


# (kind, pattern, group-with-secret)
_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)"), 0),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{22,})"), 0),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"), 0),
    ("openai_style_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-|or-v1-)?[A-Za-z0-9_\-]{20,}"), 0),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), 0),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"), 0),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"), 0),
    ("telegram_bot_token", re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_\-]{33}\b"), 0),
    ("discord_token", re.compile(r"\b[MNO][A-Za-z\d_\-]{23,27}\.[A-Za-z\d_\-]{6}\.[A-Za-z\d_\-]{27,40}\b"), 0),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), 0),
    ("url_credentials", re.compile(r"(?<=://)[^/\s:@'\"]{1,64}:([^/\s@'\"]{3,128})(?=@)"), 1),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9\-._~+/]{16,}=*)"), 1),
    (
        "quoted_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|"
            r"refresh[_-]?token|password|passwd|bot[_-]?token|token|secret)\b[\"']?\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']"
        ),
        1,
    ),
    (
        "env_assignment",
        re.compile(
            r"(?im)^\s*(?:export\s+)?[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD)[A-Z0-9_]*\s*=\s*([^\s#'\"]{8,})"
        ),
        1,
    ),
]

_GENERIC_KINDS = {"quoted_assignment", "env_assignment", "url_credentials", "bearer_token"}
_PLACEHOLDER = re.compile(
    r"(?i)^(?:x{4,}|\*{4,}|<.*>|\$\{.*\}|\$[A-Z_]+|your[_-].*|changeme.*|example.*|placeholder.*|dummy.*|"
    r"test[_-]?(?:key|token|secret|password).*|redacted.*|\[redacted\]|none|null|true|false|os\.environ.*|"
    r"process\.env.*|getenv.*|settings\..*|config\..*|env\..*)$"
)

SENSITIVE_KEY_RE = re.compile(
    r"(?i)(authorization|api[_-]?key|x-api-key|secret|token|passw|cookie|private[_-]?key|credential)"
)


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _plausible_generic(value: str) -> bool:
    if _PLACEHOLDER.match(value):
        return False
    return shannon_entropy(value) >= 3.0 and not value.isalpha()


class Redactor:
    """Thread-safe redactor combining known secret values with pattern detection."""

    def __init__(self, known_secrets: Iterable[str] = ()):
        self._lock = threading.Lock()
        self._known: set[str] = set()
        self._known_re: re.Pattern[str] | None = None
        for s in known_secrets:
            self.register_secret(s)

    def register_secret(self, value: str | None) -> None:
        if not value or len(value) < 6:
            return
        with self._lock:
            if value in self._known:
                return
            self._known.add(value)
            alternatives = sorted(self._known, key=len, reverse=True)
            self._known_re = re.compile("|".join(re.escape(v) for v in alternatives))

    def scan(self, text: str) -> list[SecretFinding]:
        findings: list[SecretFinding] = []
        if not text:
            return findings
        spans: list[tuple[int, int, str]] = []
        known = self._known_re
        if known is not None:
            spans.extend((m.start(), m.end(), "known_secret") for m in known.finditer(text))
        for kind, pattern, group in _PATTERNS:
            for m in pattern.finditer(text):
                start, end = m.span(group)
                value = m.group(group)
                if kind in _GENERIC_KINDS and not _plausible_generic(value):
                    continue
                spans.append((start, end, kind))
        spans.sort()
        last_end = -1
        for start, end, kind in spans:
            if start < last_end:
                continue
            findings.append(SecretFinding(kind, start, end, text.count("\n", 0, start) + 1))
            last_end = end
        return findings

    def redact(self, text: str) -> str:
        if not text:
            return text
        findings = self.scan(text)
        if not findings:
            return text
        out: list[str] = []
        pos = 0
        for f in findings:
            out.append(text[pos : f.start])
            out.append(f"[REDACTED:{f.kind}]")
            pos = f.end
        out.append(text[pos:])
        return "".join(out)

    def redact_obj(self, obj: Any, *, _depth: int = 0) -> Any:
        if _depth > 40:
            return obj
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            result: dict[Any, Any] = {}
            for key, value in obj.items():
                if isinstance(key, str) and SENSITIVE_KEY_RE.search(key) and isinstance(value, str) and value:
                    lowered = key.lower()
                    # Token *counts* and similar metrics are not secrets.
                    if lowered.endswith(("tokens", "_count", "token_count")) or "tokens_" in lowered:
                        result[key] = value
                    else:
                        result[key] = REDACTED
                else:
                    result[key] = self.redact_obj(value, _depth=_depth + 1)
            return result
        if isinstance(obj, (list, tuple)):
            return [self.redact_obj(v, _depth=_depth + 1) for v in obj]
        return obj


DEFAULT_REDACTOR = Redactor()
