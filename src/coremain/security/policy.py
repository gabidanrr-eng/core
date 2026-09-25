"""Central policy engine.

Every native tool call, MCP tool call, verification command and browser action is evaluated
here before it runs. Evaluation order:

1. Core invariants (cannot be overridden by config, skills, MCP servers or models): no writes
   to Core Main's own configuration/state, no paths outside the workspace, no credential
   material, no network while offline, no privilege escalation.
2. Explicit grants (scoped to project/session/task, optionally expiring or use-limited).
3. Configured rules (most restrictive matching rule wins: deny > ask > allow).
4. Permission-profile defaults (read-only / standard / autonomous), which consider the
   command's risk class and whether the workspace is isolated.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from coremain.config.schema import PermissionsConfig
from coremain.security.commands import CommandAnalysis, CommandRisk
from coremain.security.paths import is_within, sensitive_reason
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.ids import new_id

DecisionValue = Literal["allow", "ask", "deny"]
_RANK = {"allow": 0, "ask": 1, "deny": 2}


class Capability(StrEnum):
    FS_READ = "fs.read"
    FS_READ_SENSITIVE = "fs.read.sensitive"
    FS_WRITE = "fs.write"
    FS_DELETE = "fs.delete"
    EXEC = "exec.run"
    NET_HTTP = "net.http"
    NET_DOCS = "net.docs"
    BROWSER = "browser"
    GIT_READ = "git.read"
    GIT_WRITE = "git.write"
    GIT_REMOTE = "git.remote"
    EXTERNAL = "external"
    MCP = "mcp"
    MEMORY_WRITE = "memory.write"
    TASK_SPAWN = "task.spawn"
    CONFIG_WRITE = "config.write"
    SECRETS = "secrets.read"
    CONTROL = "control"  # structured-output and bookkeeping tools with no side effects


@dataclass
class PolicyRequest:
    capability: str
    target: str
    workspace_root: Path | None = None
    workspace_isolated: bool = True
    analysis: CommandAnalysis | None = None
    mcp_annotations: dict[str, Any] = field(default_factory=dict)
    mcp_trusted: bool = False
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    tool: str | None = None


@dataclass
class PolicyDecision:
    decision: DecisionValue
    reason: str
    source: str
    risk: str | None = None
    grant_id: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _domain(target: str) -> str:
    parsed = urlparse(target if "://" in target else f"http://{target}")
    return (parsed.hostname or "").lower()


def domain_matches(domain: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(domain, p.lower()) or domain == p.lower().lstrip("*.") for p in patterns)


class PolicyEngine:
    def __init__(
        self,
        config: PermissionsConfig,
        *,
        db: Database | None,
        clock: Clock,
        protected_paths: list[Path],
        profile_override: str | None = None,
    ):
        self.config = config
        self.db = db
        self.clock = clock
        self.protected_paths = [p.resolve() for p in protected_paths]
        self.profile = profile_override or config.profile

    # ------------------------------------------------------------------ public
    def evaluate(self, req: PolicyRequest) -> PolicyDecision:
        invariant = self._invariants(req)
        if invariant is not None:
            return invariant
        grant = self._grant(req)
        if grant is not None:
            return grant
        rule = self._rules(req)
        if rule is not None:
            return rule
        return self._profile_default(req)

    def describe(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "offline": self.config.network.offline,
            "rules": [r.model_dump() for r in self.config.rules],
            "allow_domains": self.config.network.allow_domains,
            "non_interactive_ask": self.config.non_interactive_ask,
        }

    # --------------------------------------------------------------- invariants
    def _protected(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == p or is_within(p, resolved) for p in self.protected_paths)

    def _invariants(self, req: PolicyRequest) -> PolicyDecision | None:
        cap = req.capability
        if cap == Capability.CONFIG_WRITE:
            return PolicyDecision(
                "deny", "models cannot change Core Main configuration or policy", "invariant"
            )
        if (
            cap in (Capability.FS_READ, Capability.FS_WRITE, Capability.FS_DELETE)
            and req.workspace_root is not None
        ):
            target = Path(req.target)
            if not target.is_absolute():
                target = req.workspace_root / target
            if not is_within(req.workspace_root, target):
                return PolicyDecision("deny", f"'{req.target}' is outside the workspace", "invariant")
            if cap != Capability.FS_READ and self._protected(target):
                return PolicyDecision(
                    "deny", "Core Main configuration, policy and state files are protected", "invariant"
                )
            reason = sensitive_reason(req.target, self.config.sensitive_paths)
            if reason and not self._has_grant(Capability.SECRETS, req):
                return PolicyDecision(
                    "deny", f"sensitive file ({reason}); grant secrets.read explicitly to allow", "invariant"
                )
        if (
            cap in (Capability.NET_HTTP, Capability.NET_DOCS, Capability.BROWSER)
            and self.config.network.offline
        ):
            domain = _domain(req.target)
            if domain not in {"localhost", "127.0.0.1", "::1", ""}:
                return PolicyDecision("deny", "offline mode: network access disabled", "invariant")
        if cap in (Capability.NET_HTTP, Capability.NET_DOCS, Capability.BROWSER):
            domain = _domain(req.target)
            if domain and domain_matches(domain, self.config.network.deny_domains):
                return PolicyDecision("deny", f"domain {domain} is denied by configuration", "invariant")
        if cap == Capability.SECRETS and not self._has_grant(Capability.SECRETS, req):
            return PolicyDecision("deny", "secret access requires an explicit user grant", "invariant")
        if cap == Capability.EXEC and req.analysis is not None:
            risks = req.analysis.risks
            if CommandRisk.SECRET_ACCESS in risks and not self._has_grant(Capability.SECRETS, req):
                return PolicyDecision(
                    "deny",
                    "command may expose secrets: " + "; ".join(req.analysis.reasons),
                    "invariant",
                    CommandRisk.SECRET_ACCESS.value,
                )
            if CommandRisk.DESTRUCTIVE in risks and any(
                "outside the workspace" in r for r in req.analysis.reasons
            ):
                return PolicyDecision(
                    "deny",
                    "destructive operation outside the workspace",
                    "invariant",
                    CommandRisk.DESTRUCTIVE.value,
                )
            if self.config.network.offline and CommandRisk.NETWORK in risks:
                return PolicyDecision(
                    "deny",
                    "offline mode: command performs network access",
                    "invariant",
                    CommandRisk.NETWORK.value,
                )
            if req.workspace_root is not None and any(
                self._mentions_protected(a) for a in req.target.split()
            ):
                return PolicyDecision("deny", "command references protected Core Main state", "invariant")
        return None

    def _mentions_protected(self, token: str) -> bool:
        if "/" not in token:
            return False
        try:
            return self._protected(Path(token.strip("'\"")).expanduser())
        except (OSError, ValueError):
            return False

    # ------------------------------------------------------------------- grants
    def _grant_rows(self, capability: str, req: PolicyRequest) -> list[Any]:
        if self.db is None:
            return []
        now = self.clock.now()
        return self.db.query(
            "SELECT * FROM permission_grants WHERE capability = ? AND revoked_at IS NULL "
            "AND (expires_at IS NULL OR expires_at > ?) AND (uses_remaining IS NULL OR uses_remaining > 0) "
            "AND (project_id IS NULL OR project_id = ?) AND (session_id IS NULL OR session_id = ?) "
            "AND (task_id IS NULL OR task_id = ?) ORDER BY created_at DESC",
            (capability, now, req.project_id, req.session_id, req.task_id),
        )

    def _has_grant(self, capability: str, req: PolicyRequest) -> bool:
        return any(
            r["decision"] == "allow" and fnmatch.fnmatch(req.target, r["pattern"])
            for r in self._grant_rows(capability, req)
        )

    def _grant(self, req: PolicyRequest) -> PolicyDecision | None:
        rows = [r for r in self._grant_rows(req.capability, req) if fnmatch.fnmatch(req.target, r["pattern"])]
        if not rows:
            return None
        deny = next((r for r in rows if r["decision"] == "deny"), None)
        chosen = deny or rows[0]
        if (
            chosen["decision"] == "allow"
            and req.analysis is not None
            and CommandRisk.PRIVILEGED in req.analysis.risks
            and chosen["pattern"] == "*"
        ):
            return None  # blanket grants never cover privileged commands; a specific pattern is required
        if chosen["uses_remaining"] is not None and self.db is not None:
            self.db.execute(
                "UPDATE permission_grants SET uses_remaining = uses_remaining - 1 WHERE id = ?",
                (chosen["id"],),
            )
        return PolicyDecision(
            chosen["decision"],
            chosen["reason"] or f"granted by {chosen['granted_by']}",
            "grant",
            grant_id=chosen["id"],
        )

    def grant(
        self,
        capability: str,
        *,
        pattern: str = "*",
        decision: Literal["allow", "deny"] = "allow",
        granted_by: str = "user",
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        ttl_s: float | None = None,
        uses: int | None = None,
        reason: str | None = None,
    ) -> str:
        if self.db is None:
            raise RuntimeError("grants require a database")
        if granted_by not in {"user", "approval"}:
            raise ValueError("only users (directly or via approvals) can create grants")
        now = self.clock.now()
        grant_id = new_id("grt", now=now)
        self.db.execute(
            "INSERT INTO permission_grants(id, project_id, session_id, task_id, capability, pattern, decision, granted_by, reason, "
            "created_at, expires_at, uses_remaining) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                grant_id,
                project_id,
                session_id,
                task_id,
                capability,
                pattern,
                decision,
                granted_by,
                reason,
                now,
                now + ttl_s if ttl_s else None,
                uses,
            ),
        )
        return grant_id

    def revoke(self, grant_id: str) -> bool:
        if self.db is None:
            return False
        cur = self.db.execute(
            "UPDATE permission_grants SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (self.clock.now(), grant_id),
        )
        return cur.rowcount == 1

    def active_grants(self, project_id: str | None = None) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        now = self.clock.now()
        rows = self.db.query(
            "SELECT * FROM permission_grants WHERE revoked_at IS NULL AND (expires_at IS NULL OR expires_at > ?) "
            "AND (uses_remaining IS NULL OR uses_remaining > 0) AND (project_id IS NULL OR project_id = ?) ORDER BY created_at DESC",
            (now, project_id),
        )
        return [dict(r) for r in rows]

    # -------------------------------------------------------------------- rules
    def _rules(self, req: PolicyRequest) -> PolicyDecision | None:
        best: PolicyDecision | None = None
        for rule in self.config.rules:
            if not fnmatch.fnmatch(req.capability, rule.capability):
                continue
            target = (
                _domain(req.target)
                if req.capability in (Capability.NET_HTTP, Capability.NET_DOCS) and "/" not in rule.pattern
                else req.target
            )
            if not fnmatch.fnmatch(target, rule.pattern):
                continue
            candidate = PolicyDecision(
                rule.decision, rule.reason or f"rule {rule.capability}:{rule.pattern}", "rule"
            )
            if best is None or _RANK[candidate.decision] > _RANK[best.decision]:
                best = candidate
        if (
            best is not None
            and best.decision == "allow"
            and req.analysis is not None
            and CommandRisk.PRIVILEGED in req.analysis.risks
            and not any(
                r.decision == "allow" and r.pattern != "*" and fnmatch.fnmatch(req.target, r.pattern)
                for r in self.config.rules
            )
        ):
            return None
        return best

    # ------------------------------------------------------------ profile defaults
    def _profile_default(self, req: PolicyRequest) -> PolicyDecision:
        cap = req.capability
        profile = self.profile
        iso = req.workspace_isolated

        def d(value: DecisionValue, reason: str, risk: str | None = None) -> PolicyDecision:
            return PolicyDecision(value, reason, f"profile:{profile}", risk)

        if cap in (Capability.FS_READ, Capability.GIT_READ, Capability.CONTROL, Capability.MEMORY_WRITE):
            return d("allow", f"{cap} is permitted")
        if cap == Capability.TASK_SPAWN:
            return d("allow", "bounded subtask decomposition")
        if cap == Capability.FS_READ_SENSITIVE:
            return d("deny", "sensitive files require an explicit grant")
        if profile == "read-only":
            if (
                cap == Capability.EXEC
                and req.analysis is not None
                and req.analysis.risks <= {CommandRisk.READ_ONLY, CommandRisk.GIT_READ}
            ):
                return d("allow", "read-only command", req.analysis.max_risk.value)
            if cap == Capability.NET_DOCS and domain_matches(
                _domain(req.target), self.config.network.allow_domains
            ):
                return d("allow", "documentation lookup on an allowed domain")
            if cap == Capability.MCP and req.mcp_trusted and req.mcp_annotations.get("readOnlyHint"):
                return d("allow", "read-only tool on a trusted MCP server")
            return d("deny", "read-only permission profile")
        if cap == Capability.FS_WRITE:
            return d(
                "allow", "local file change inside the task workspace" + ("" if iso else " (direct mode)")
            )
        if cap == Capability.FS_DELETE:
            return d(
                "allow" if iso else "ask", "file deletion" + ("" if iso else " in the canonical workspace")
            )
        if cap == Capability.GIT_WRITE:
            return d("allow" if iso or profile == "autonomous" else "ask", "local git mutation")
        if cap == Capability.GIT_REMOTE:
            return d(
                "ask" if profile == "autonomous" else "deny",
                "remote git operations need explicit authorization",
            )
        if cap == Capability.EXTERNAL:
            return d(
                "ask" if profile == "autonomous" else "deny",
                "external side effects need explicit authorization",
            )
        if cap in (Capability.NET_DOCS, Capability.NET_HTTP):
            domain = _domain(req.target)
            if domain_matches(domain, self.config.network.allow_domains) or profile == "autonomous":
                return d("allow", f"network access to {domain}")
            return d("ask", f"network access to {domain} (not in allow_domains)")
        if cap == Capability.BROWSER:
            domain = _domain(req.target)
            if domain in {"localhost", "127.0.0.1", "::1", ""} or profile == "autonomous":
                return d("allow", "browser automation on a local target")
            return d("ask", f"browser navigation to {domain}")
        if cap == Capability.MCP:
            ann = req.mcp_annotations
            if ann.get("destructiveHint") and not ann.get("readOnlyHint"):
                return d("ask", "destructive MCP tool")
            if req.mcp_trusted and (ann.get("readOnlyHint") or profile == "autonomous"):
                return d("allow", "trusted MCP server")
            if ann.get("readOnlyHint") and not ann.get("openWorldHint"):
                return d("allow", "read-only MCP tool")
            return d(
                "ask",
                "MCP tool with possible side effects"
                + ("" if req.mcp_trusted else " on an untrusted server"),
            )
        if cap == Capability.EXEC:
            return self._exec_default(req, profile, iso)
        return d("ask", f"no default for capability {cap}")

    def _exec_default(self, req: PolicyRequest, profile: str, iso: bool) -> PolicyDecision:
        analysis = req.analysis
        src = f"profile:{profile}"
        if analysis is None:
            return PolicyDecision("ask", "unanalyzed command", src)
        risks = analysis.risks
        top = analysis.max_risk.value
        why = "; ".join(analysis.reasons[:3])
        if CommandRisk.PRIVILEGED in risks:
            return PolicyDecision("deny", f"privileged command ({why})", src, top)
        if CommandRisk.EXTERNAL in risks:
            return PolicyDecision(
                "ask" if profile == "autonomous" else "deny", f"external side effect ({why})", src, top
            )
        if CommandRisk.DESTRUCTIVE in risks:
            if profile == "autonomous" and iso:
                return PolicyDecision(
                    "allow", f"destructive but confined to an isolated workspace ({why})", src, top
                )
            return PolicyDecision("ask" if iso else "deny", f"destructive command ({why})", src, top)
        if profile == "autonomous":
            return PolicyDecision("allow", f"autonomous profile ({why})", src, top)
        if risks & {CommandRisk.PACKAGE_INSTALL, CommandRisk.NETWORK, CommandRisk.UNKNOWN}:
            return PolicyDecision("ask", why, src, top)
        if not iso and risks & {CommandRisk.LOCAL_MUTATION, CommandRisk.GIT_WRITE}:
            return PolicyDecision("ask", f"mutation of the canonical workspace via shell ({why})", src, top)
        return PolicyDecision("allow", why, src, top)
