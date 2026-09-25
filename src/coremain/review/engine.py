"""Review engine: persists reviews and findings bound to the exact diff under review."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coremain.events import EventLog
from coremain.review.static_checks import Finding
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id

BLOCKING = {"critical", "high"}


@dataclass
class ReviewRecord:
    id: str
    strategy: str
    reviewer: str
    verdict: str
    summary: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    independence: str | None = None


class ReviewStore:
    def __init__(self, db: Database, events: EventLog, clock: Clock):
        self.db = db
        self.events = events
        self.clock = clock

    def record(
        self,
        *,
        project_id: str,
        task_id: str,
        attempt_id: str | None,
        strategy: str,
        reviewer: str,
        verdict: str,
        summary: str,
        findings: list[Finding],
        diff_hash: str,
        fingerprint: str | None,
        workspace_root: Path | None,
        independence: str | None = None,
    ) -> ReviewRecord:
        now = self.clock.now()
        review_id = new_id("rev", now=now)
        stored: list[dict[str, Any]] = []
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO reviews(id, task_id, attempt_id, strategy, reviewer, independence, diff_hash, fingerprint, verdict, summary, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    review_id,
                    task_id,
                    attempt_id,
                    strategy,
                    reviewer,
                    independence,
                    diff_hash,
                    fingerprint,
                    verdict,
                    summary[:2000],
                    now,
                ),
            )
            for f in findings:
                valid = None
                if f.file and workspace_root is not None:
                    target = workspace_root / f.file
                    valid = 1 if target.is_file() else 0
                    if valid and f.line:
                        try:
                            valid = (
                                1
                                if f.line
                                <= len(target.read_text(encoding="utf-8", errors="replace").splitlines()) + 1
                                else 0
                            )
                        except OSError:
                            valid = 0
                finding_id = new_id("fnd", now=now)
                conn.execute(
                    "INSERT INTO findings(id, task_id, attempt_id, review_id, severity, category, title, file, line, rationale, evidence, "
                    "remediation, status, diff_hash, location_valid, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        finding_id,
                        task_id,
                        attempt_id,
                        review_id,
                        f.severity,
                        f.category,
                        f.title[:300],
                        f.file,
                        f.line,
                        f.rationale[:2000],
                        f.evidence[:2000],
                        f.remediation[:1000],
                        "open",
                        diff_hash,
                        valid,
                        now,
                    ),
                )
                stored.append({**f.to_dict(), "id": finding_id, "location_valid": valid})
            self.events.emit(
                "review.completed",
                project_id=project_id,
                task_id=task_id,
                attempt_id=attempt_id,
                level="info" if verdict == "approve" else "warning",
                data={
                    "review_id": review_id,
                    "strategy": strategy,
                    "reviewer": reviewer,
                    "verdict": verdict,
                    "summary": summary[:300],
                    "findings": [
                        {"severity": s["severity"], "title": s["title"], "file": s["file"], "line": s["line"]}
                        for s in stored
                    ][:30],
                    "independence": independence,
                },
            )
        return ReviewRecord(review_id, strategy, reviewer, verdict, summary, stored, independence)

    def supersede_older(self, task_id: str, diff_hash: str) -> int:
        cur = self.db.execute(
            "UPDATE findings SET status = 'resolved', resolution = ?, resolved_at = ? WHERE task_id = ? AND status = 'open' AND diff_hash <> ?",
            (f"superseded by review of diff {diff_hash[:12]}", self.clock.now(), task_id, diff_hash),
        )
        return cur.rowcount

    def open_findings(self, task_id: str, *, diff_hash: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM findings WHERE task_id = ? AND status = 'open'"
        params: list[Any] = [task_id]
        if diff_hash:
            sql += " AND diff_hash = ?"
            params.append(diff_hash)
        return [
            dict(r)
            for r in self.db.query(
                sql
                + " ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 "
                "WHEN 'low' THEN 3 ELSE 4 END, created_at",
                params,
            )
        ]

    def reviews(self, task_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.query("SELECT * FROM reviews WHERE task_id = ? ORDER BY created_at", (task_id,))
        ]

    def dismiss(self, finding_id: str, reason: str) -> None:
        self.db.execute(
            "UPDATE findings SET status = 'dismissed', resolution = ?, resolved_at = ? WHERE id = ?",
            (f"dismissed by user: {reason}", self.clock.now(), finding_id),
        )


def verdict_for(findings: list[dict[str, Any]], model_verdicts: list[str]) -> str:
    if any(f["severity"] in BLOCKING for f in findings):
        return "request_changes"
    if "request_changes" in model_verdicts and any(f["severity"] in {"medium"} for f in findings):
        return "request_changes"
    if model_verdicts and all(v == "inconclusive" for v in model_verdicts):
        return "inconclusive"
    return "approve"
