"""Time-sortable identifiers (ULID layout, lowercase Crockford base32) with a type prefix."""

from __future__ import annotations

import secrets
import time

_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"

PREFIXES = {
    "project": "prj",
    "session": "ses",
    "message": "msg",
    "task": "tsk",
    "attempt": "att",
    "workspace": "ws",
    "artifact": "art",
    "evidence": "evd",
    "model_call": "mc",
    "tool_call": "tc",
    "process": "prc",
    "approval": "apr",
    "grant": "grt",
    "context": "ctx",
    "review": "rev",
    "finding": "fnd",
    "memory": "mem",
    "decision": "adr",
    "failure": "fail",
    "regression": "reg",
    "heuristic": "heu",
    "hypothesis": "hyp",
    "checkpoint": "chk",
    "runtime": "rt",
    "eval_run": "evr",
    "eval_result": "evx",
}


def new_id(prefix: str, *, now: float | None = None) -> str:
    ts_ms = int((time.time() if now is None else now) * 1000) & ((1 << 48) - 1)
    value = (ts_ms << 80) | secrets.randbits(80)
    chars = []
    for _ in range(26):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return f"{prefix}_{''.join(reversed(chars))}"


def short_id(identifier: str, length: int = 8) -> str:
    """Human-friendly short form: ``tsk_…abcd1234`` → ``tsk_abcd1234`` (random tail, unique in practice)."""
    if "_" not in identifier:
        return identifier[-length:]
    prefix, body = identifier.split("_", 1)
    return f"{prefix}_{body[-length:]}"
