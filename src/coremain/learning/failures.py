"""Structured failure records with normalized signatures so recurrences can be recognized."""

from __future__ import annotations

import re
from typing import Any

from coremain.events import EventLog
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, sha256_hex

_NORMALIZE = [
    (re.compile(r"0x[0-9a-f]+", re.I), "<hex>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<id>"),
    (re.compile(r"(/[\w.@-]+)+"), "<path>"),
    (re.compile(r"\b\d+(\.\d+)?\b"), "<n>"),
    (re.compile(r"'[^']*'|\"[^\"]*\""), "<str>"),
    (re.compile(r"\s+"), " "),
]


def failure_signature(category: str, error_class: str, summary: str) -> str:
    text = summary.lower()
    for rx, repl in _NORMALIZE:
        text = rx.sub(repl, text)
    return sha256_hex(f"{category}|{error_class}|{text.strip()[:300]}")[:24]


class FailureRecorder:
    def __init__(self, db: Database, events: EventLog, clock: Clock):
        self.db = db
        self.events = events
        self.clock = clock

    def record(self, *, category: str, error_class: str, summary: str, project_id: str | None = None, task_id: str | None = None,
               attempt_id: str | None = None, stage: str | None = None, context: dict[str, Any] | None = None) -> str:
        now = self.clock.now()
        failure_id = new_id("fail", now=now)
        signature = failure_signature(category, error_class, summary)
        self.db.execute(
            "INSERT INTO failures(id, project_id, task_id, attempt_id, stage, category, error_class, signature, summary, context_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (failure_id, project_id, task_id, attempt_id, stage, category, error_class, signature, summary[:2000], dumps(context or {}), now),
        )
        recurrences = int(self.db.scalar("SELECT COUNT(*) FROM failures WHERE signature = ?", (signature,)) or 0)
        self.events.emit("failure.recorded", project_id=project_id, task_id=task_id, attempt_id=attempt_id, level="warning",
                         data={"failure_id": failure_id, "category": category, "error_class": error_class, "signature": signature,
                               "recurrences": recurrences, "summary": summary[:300]})
        return failure_id

    def resolve_for_task(self, task_id: str, *, resolution: str, evidence_id: str | None, verified: bool) -> int:
        now = self.clock.now()
        cur = self.db.execute(
            "UPDATE failures SET outcome = 'resolved', resolution = ?, resolution_evidence_id = ?, verified = ?, resolved_at = ?, "
            "regression_candidate = CASE WHEN category IN ('verification', 'review', 'tool', 'workflow') THEN 1 ELSE regression_candidate END "
            "WHERE task_id = ? AND outcome = 'open'",
            (resolution[:1000], evidence_id, 1 if verified else 0, now, task_id),
        )
        return cur.rowcount

    def for_task(self, task_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM failures WHERE task_id = ? ORDER BY created_at", (task_id,))]
