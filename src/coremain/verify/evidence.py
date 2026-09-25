"""Evidence engine.

Evidence is the basis for truth; model output is only a claim. Each evidence record carries
its provenance (task, attempt, workspace, command/tool, provider/model) and the workspace
*fingerprint* (content tree id) at the moment it was produced. The gate only accepts evidence
that belongs to the same task, attempt and workspace and whose fingerprint equals the current
one, so results from another attempt, another workspace or an earlier state of the code can
never prove completion. Missing, stale, failing or contradictory evidence fails closed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from coremain.domain.models import GateResult
from coremain.events import EventLog
from coremain.runtime.leases import Fence, LeaseManager
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id
from coremain.util.jsonutil import dumps, loads


class EvidenceKind(StrEnum):
    TESTS = "tests"
    LINT = "lint"
    TYPECHECK = "typecheck"
    BUILD = "build"
    FORMAT = "format"
    SYNTAX = "syntax"
    DIFF = "diff"
    NO_SECRETS = "no_secrets"
    SCOPE = "scope"
    WORKSPACE_UNCHANGED = "workspace_unchanged"
    WORKSPACE_BINDING = "workspace_binding"
    ANSWER = "answer"
    CITATIONS = "citations"
    REVIEW = "review"
    BROWSER = "browser"
    COMMAND = "command"
    REPRO = "repro"
    MODEL_CLAIM = "model_claim"
    APPLY = "apply"
    GATE = "gate"


class Trust(StrEnum):
    VERIFIED = "verified"  # computed deterministically by the runtime (diffs, scans, records)
    OBSERVED = "observed"  # the runtime executed something and observed the outcome
    CLAIMED = "claimed"  # a model asserted it; never sufficient on its own
    EXTERNAL = "external"  # retrieved from outside the repository


ACCEPTED_TRUST = (Trust.VERIFIED.value, Trust.OBSERVED.value)
LEVELS = ("none", "weak", "moderate", "strong")


@dataclass
class Requirement:
    kind: str
    name: str = ""
    description: str = ""
    state_bound: bool = True
    required: bool = True

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}" if self.name else self.kind

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Requirement:
        return cls(d["kind"], d.get("name", ""), d.get("description", ""), d.get("state_bound", True), d.get("required", True))


@dataclass
class Evidence:
    id: str
    project_id: str
    task_id: str
    attempt_id: str | None
    workspace_id: str | None
    kind: str
    status: str
    trust: str
    summary: str
    fingerprint: str | None
    diff_hash: str | None
    command: str | None
    exit_code: int | None
    tool: str | None
    provider: str | None
    model: str | None
    artifact_id: str | None
    data: dict[str, Any]
    created_at: float

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Evidence:
        return cls(r["id"], r["project_id"], r["task_id"], r["attempt_id"], r["workspace_id"], r["kind"], r["status"], r["trust"],
                   r["summary"], r["fingerprint"], r["diff_hash"], r["command"], r["exit_code"], r["tool"], r["provider"],
                   r["model"], r["artifact_id"], loads(r["data_json"], {}), r["created_at"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GateContext:
    task_id: str
    attempt_id: str
    workspace_id: str | None
    fingerprint: str | None
    requirements: list[Requirement]
    min_level: str = "weak"
    diff_hash: str | None = None
    notes: list[str] = field(default_factory=list)


def evidence_level(satisfied_kinds: set[str]) -> str:
    if EvidenceKind.TESTS in satisfied_kinds:
        return "strong"
    if satisfied_kinds & {EvidenceKind.BUILD, EvidenceKind.TYPECHECK, EvidenceKind.BROWSER, EvidenceKind.COMMAND}:
        return "moderate"
    if satisfied_kinds & {EvidenceKind.SYNTAX, EvidenceKind.LINT, EvidenceKind.REVIEW, EvidenceKind.CITATIONS,
                          EvidenceKind.WORKSPACE_UNCHANGED, EvidenceKind.DIFF}:
        return "weak"
    return "none"


class EvidenceEngine:
    def __init__(self, db: Database, events: EventLog, leases: LeaseManager, clock: Clock):
        self.db = db
        self.events = events
        self.leases = leases
        self.clock = clock

    def record(
        self,
        *,
        project_id: str,
        task_id: str,
        kind: str,
        status: str,
        trust: str,
        summary: str,
        attempt_id: str | None = None,
        workspace_id: str | None = None,
        fingerprint: str | None = None,
        diff_hash: str | None = None,
        command: str | None = None,
        exit_code: int | None = None,
        tool: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        artifact_id: str | None = None,
        data: dict[str, Any] | None = None,
        fence: Fence | None = None,
    ) -> Evidence:
        now = self.clock.now()
        evidence_id = new_id("evd", now=now)
        with self.db.tx() as conn:
            if fence is not None:
                self.leases.verify(conn, fence)
            conn.execute(
                "INSERT INTO evidence(id, project_id, task_id, attempt_id, workspace_id, kind, status, trust, summary, fingerprint, "
                "diff_hash, command, exit_code, tool, provider, model, artifact_id, data_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (evidence_id, project_id, task_id, attempt_id, workspace_id, kind, status, trust, summary[:2000], fingerprint,
                 diff_hash, command, exit_code, tool, provider, model, artifact_id, dumps(data or {}), now),
            )
            self.events.emit(
                "evidence.recorded", project_id=project_id, task_id=task_id, attempt_id=attempt_id,
                level="info" if status == "pass" else "warning",
                data={"evidence_id": evidence_id, "kind": kind, "status": status, "trust": trust, "summary": summary[:300],
                      "name": (data or {}).get("name")},
            )
            row = conn.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
        return Evidence.from_row(row)

    def for_task(self, task_id: str, *, attempt_id: str | None = None, kinds: tuple[str, ...] | None = None) -> list[Evidence]:
        sql = "SELECT * FROM evidence WHERE task_id = ?"
        params: list[Any] = [task_id]
        if attempt_id:
            sql += " AND attempt_id = ?"
            params.append(attempt_id)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        return [Evidence.from_row(r) for r in self.db.query(sql + " ORDER BY created_at, rowid", params)]

    def evaluate(self, conn: sqlite3.Connection, ctx: GateContext) -> GateResult:
        """Evaluate requirements against evidence bound to this task/attempt/workspace/state."""
        rows = conn.execute(
            "SELECT * FROM evidence WHERE task_id = ? AND attempt_id = ? ORDER BY created_at, rowid",
            (ctx.task_id, ctx.attempt_id),
        ).fetchall()
        evidence = [Evidence.from_row(r) for r in rows]
        satisfied: list[str] = []
        missing: list[str] = []
        failing: list[str] = []
        stale: list[str] = []
        notes = list(ctx.notes)
        satisfied_kinds: set[str] = set()
        for req in ctx.requirements:
            matches = [
                e for e in evidence
                if e.kind == req.kind and (not req.name or e.data.get("name") == req.name) and e.trust in ACCEPTED_TRUST
            ]
            foreign = [e for e in matches if ctx.workspace_id is not None and e.workspace_id not in (None, ctx.workspace_id)]
            if foreign:
                notes.append(f"{req.key}: ignored {len(foreign)} record(s) from another workspace")
            matches = [e for e in matches if e not in foreign]
            claimed = [e for e in evidence if e.kind == req.kind and e.trust == Trust.CLAIMED]
            if not matches:
                (missing if req.required else notes).append(
                    req.key if req.required else f"{req.key}: optional evidence missing"
                )
                if claimed:
                    notes.append(f"{req.key}: only model claims present (not accepted as proof)")
                continue
            current = [e for e in matches if not req.state_bound or e.fingerprint == ctx.fingerprint]
            if not current:
                if req.required:
                    stale.append(req.key)
                else:
                    notes.append(f"{req.key}: optional evidence is stale")
                continue
            latest = current[-1]
            contradictory = req.state_bound and any(e.status == "fail" for e in current) and latest.status == "pass"
            if contradictory:
                notes.append(f"{req.key}: contradictory results for the identical workspace state (flaky?)")
            if latest.status != "pass" or contradictory:
                if req.required:
                    failing.append(req.key)
                else:
                    notes.append(f"{req.key}: optional check did not pass ({latest.status})")
                continue
            satisfied.append(req.key)
            satisfied_kinds.add(req.kind)
        level = evidence_level(satisfied_kinds)
        passed = not (missing or failing or stale) and LEVELS.index(level) >= LEVELS.index(ctx.min_level)
        if not (missing or failing or stale) and not passed:
            notes.append(f"evidence level '{level}' is below the required minimum '{ctx.min_level}'")
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if failing:
            parts.append("failing " + ", ".join(failing))
        if stale:
            parts.append("stale " + ", ".join(stale))
        summary = "; ".join(parts) if parts else (f"all {len(satisfied)} requirement(s) satisfied at level {level}" if passed else notes[-1])
        return GateResult(passed, summary, level, satisfied, missing, failing, stale, notes)
