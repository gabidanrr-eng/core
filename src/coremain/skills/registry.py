"""Skill discovery, validation, trust and selection.

Format: a directory containing ``SKILL.md`` with YAML frontmatter (``name``, ``description``,
optional ``license``, ``compatibility``, ``metadata`` string map, ``allowed-tools``) following
the Agent Skills specification. Core Main extensions live in ``metadata`` under ``core-*``
keys (triggers, kinds, languages, frameworks, stage, priority, requires-tools, conflicts,
version) so skills remain portable.

Trust: built-in skills ship with Core Main; user skills (in the user config dir) are trusted;
project and imported skills are untrusted until ``core skills trust``, and trust is pinned to
the directory's content hash, so any change revokes it. Skills provide guidance only: they
cannot grant permissions or alter policy, which the runtime enforces independently.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from coremain.config.schema import SkillsConfig
from coremain.errors import CoreError, NotFoundError
from coremain.events import EventLog
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.jsonutil import sha256_hex
from coremain.util.text import estimate_tokens, query_terms

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PROVENANCE_FILE = ".core-provenance.json"
SOURCE_PRECEDENCE = {"builtin": 0, "imported": 1, "user": 2, "project": 3}


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    source: str
    body: str
    frontmatter: dict[str, Any]
    sha256: str
    metadata: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trust: str = "untrusted"
    resources: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.problems

    @property
    def usable(self) -> bool:
        return self.valid and self.trust in {"builtin", "trusted"}

    def meta_list(self, key: str) -> list[str]:
        raw = self.metadata.get(key, "")
        return [x.strip().lower() for x in raw.split(",") if x.strip()]

    @property
    def version(self) -> str:
        return self.metadata.get("core-version", "0")

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.body)

    def to_dict(self, *, include_body: bool = False) -> dict[str, Any]:
        d = {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "source": self.source,
            "sha256": self.sha256,
            "version": self.version,
            "trust": self.trust,
            "valid": self.valid,
            "problems": self.problems,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "resources": self.resources,
            "tokens": self.tokens,
            "provenance": self.provenance,
            "license": self.frontmatter.get("license"),
            "allowed_tools": self.frontmatter.get("allowed-tools"),
        }
        if include_body:
            d["body"] = self.body
        return d


@dataclass
class SkillSelection:
    skill: Skill
    score: float
    reasons: list[str]


def dir_hash(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.name != PROVENANCE_FILE and "__pycache__" not in p.parts
    ):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def parse_skill_md(text: str) -> tuple[dict[str, Any], str, list[str]]:
    problems: list[str] = []
    if not text.startswith("---"):
        return {}, text, ["SKILL.md must start with YAML frontmatter delimited by ---"]
    parts = text.split("\n---", 1)
    if len(parts) != 2:
        return {}, text, ["unterminated frontmatter"]
    raw_front = parts[0][3:]
    body = parts[1].lstrip("-").lstrip("\n")
    try:
        front = yaml.safe_load(raw_front) or {}
    except yaml.YAMLError as exc:
        return {}, body, [f"frontmatter is not valid YAML: {exc.__class__.__name__}"]
    if not isinstance(front, dict):
        return {}, body, ["frontmatter must be a mapping"]
    return front, body, problems


def load_skill(path: Path, source: str) -> Skill:
    md = path / "SKILL.md"
    text = md.read_text(encoding="utf-8", errors="replace")
    front, body, problems = parse_skill_md(text)
    warnings: list[str] = []
    name = str(front.get("name", "")).strip()
    description = str(front.get("description", "")).strip()
    if not name:
        problems.append("missing required 'name'")
    elif not (1 <= len(name) <= 64) or not NAME_RE.match(name):
        problems.append("'name' must be 1-64 chars of a-z, 0-9 and single hyphens")
    elif name != path.name:
        warnings.append(f"name '{name}' does not match directory '{path.name}'")
    if not description:
        problems.append("missing required 'description'")
    elif len(description) > 1024:
        warnings.append("description exceeds 1024 characters")
    meta_raw = front.get("metadata", {}) or {}
    metadata: dict[str, str] = {}
    if isinstance(meta_raw, dict):
        metadata = {str(k): str(v) for k, v in meta_raw.items()}
    else:
        warnings.append("'metadata' should be a string→string map")
    compat = front.get("compatibility")
    if compat is not None and not (1 <= len(str(compat)) <= 500):
        warnings.append("'compatibility' should be 1-500 characters")
    if estimate_tokens(body) > 6000:
        warnings.append("body is large (>6000 tokens); consider moving detail into references/")
    resources_list = sorted(
        p.relative_to(path).as_posix()
        for p in path.rglob("*")
        if p.is_file() and p.name not in {"SKILL.md", PROVENANCE_FILE} and "__pycache__" not in p.parts
    )
    provenance: dict[str, Any] = {}
    prov_path = path / PROVENANCE_FILE
    if prov_path.exists():
        try:
            provenance = json.loads(prov_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warnings.append("unreadable provenance file")
    return Skill(
        name or path.name,
        description,
        path,
        source,
        body,
        front,
        dir_hash(path),
        metadata,
        problems,
        warnings,
        resources=resources_list,
        provenance=provenance,
    )


def builtin_skills_dir() -> Path:
    return Path(str(resources.files("coremain.skills") / "builtin"))


class SkillRegistry:
    def __init__(
        self,
        db: Database | None,
        config: SkillsConfig,
        roots: list[tuple[str, Path]],
        clock: Clock,
        events: EventLog | None = None,
    ):
        self.db = db
        self.config = config
        self.roots = roots
        self.clock = clock
        self.events = events
        self._skills: dict[str, Skill] | None = None
        self.collisions: list[str] = []

    def refresh(self) -> dict[str, Skill]:
        found: dict[str, Skill] = {}
        self.collisions = []
        for source, root in sorted(self.roots, key=lambda r: SOURCE_PRECEDENCE.get(r[0], 9)):
            if not root.is_dir():
                continue
            for md in sorted(root.glob("*/SKILL.md")) + sorted(root.glob("*/*/SKILL.md")):
                skill = load_skill(md.parent, source)
                if skill.name in found and found[skill.name].path != skill.path:
                    self.collisions.append(
                        f"{skill.name}: {source} skill at {skill.path} overrides {found[skill.name].source} skill"
                    )
                    skill.warnings.append(f"overrides {found[skill.name].source} skill of the same name")
                found[skill.name] = skill
        for skill in found.values():
            skill.trust = self._trust_for(skill)
        for skill in found.values():
            for other in skill.meta_list("core-conflicts"):
                if other in found:
                    skill.warnings.append(
                        f"declares a conflict with '{other}' (both installed; never selected together)"
                    )
        self._skills = found
        return found

    def _trust_for(self, skill: Skill) -> str:
        if skill.name in self.config.disabled:
            return "disabled"
        decision = None
        if self.db is not None:
            row = self.db.one(
                "SELECT decision FROM skill_trust WHERE name = ? AND sha256 = ?", (skill.name, skill.sha256)
            )
            decision = row["decision"] if row else None
        if decision == "blocked":
            return "blocked"
        if skill.source == "builtin":
            return "builtin"
        if decision == "trusted" or skill.source == "user":
            return "trusted"
        if self.db is not None and self.db.one(
            "SELECT 1 FROM skill_trust WHERE name = ? AND decision = 'trusted'", (skill.name,)
        ):
            skill.warnings.append("content changed since it was trusted; re-run `core skills trust`")
        return "untrusted"

    def all(self) -> list[Skill]:
        skills = self._skills if self._skills is not None else self.refresh()
        return sorted(skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill:
        skills = self._skills if self._skills is not None else self.refresh()
        if name not in skills:
            raise NotFoundError(f"skill '{name}' not found", hint="List skills with `core skills list`.")
        return skills[name]

    def set_trust(self, name: str, decision: str, *, actor: str = "user") -> Skill:
        skill = self.get(name)
        if self.db is None:
            raise CoreError("skill trust requires the database")
        if decision == "untrusted":
            self.db.execute("DELETE FROM skill_trust WHERE name = ?", (name,))
        else:
            self.db.execute(
                "INSERT OR REPLACE INTO skill_trust(name, sha256, decision, source, decided_at) VALUES (?,?,?,?,?)",
                (name, skill.sha256, decision, str(skill.path), self.clock.now()),
            )
        if self.events is not None:
            self.events.emit(
                "skill.trust", actor=actor, data={"skill": name, "decision": decision, "sha256": skill.sha256}
            )
        self.refresh()
        return self.get(name)

    def select(
        self,
        *,
        text: str,
        task_kind: str,
        languages: list[str],
        frameworks: list[str],
        stage: str,
        max_n: int | None = None,
        max_tokens: int | None = None,
    ) -> list[SkillSelection]:
        if not self.config.enabled:
            return []
        max_n = self.config.max_selected if max_n is None else max_n
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        lowered = text.lower()
        terms = set(query_terms(text, max_terms=40))
        langs = {x.lower() for x in languages}
        fws = {x.lower() for x in frameworks}
        candidates: list[SkillSelection] = []
        for skill in self.all():
            if not skill.valid or not (
                skill.usable or (self.config.allow_untrusted and skill.trust == "untrusted")
            ):
                continue
            score = 0.0
            reasons: list[str] = []
            hits = [t for t in skill.meta_list("core-triggers") if t and t in lowered]
            if hits:
                score += 2.0 * min(3, len(hits))
                reasons.append("triggers: " + ", ".join(hits[:3]))
            if task_kind.lower() in skill.meta_list("core-kinds"):
                score += 3.0
                reasons.append(f"task kind {task_kind}")
            stages = skill.meta_list("core-stage")
            if stages and stage.lower() not in stages and "any" not in stages:
                score -= 2.0
            elif stages and stage.lower() in stages:
                score += 0.5
            lang_hits = langs & set(skill.meta_list("core-languages"))
            fw_hits = fws & set(skill.meta_list("core-frameworks"))
            if lang_hits or fw_hits:
                score += 1.0 * len(lang_hits | fw_hits)
                reasons.append("stack: " + ", ".join(sorted(lang_hits | fw_hits)))
            desc_terms = set(query_terms(skill.description, max_terms=60))
            overlap = len(terms & desc_terms)
            if overlap:
                score += 0.3 * min(overlap, 6)
            try:
                score += min(1.0, float(skill.metadata.get("core-priority", "0")) / 10)
            except ValueError:
                pass
            if score >= 2.5:
                candidates.append(SkillSelection(skill, score, reasons or ["description relevance"]))
        candidates.sort(key=lambda s: s.score, reverse=True)
        chosen: list[SkillSelection] = []
        used = 0
        for cand in candidates:
            if len(chosen) >= max_n:
                break
            conflicts = set(cand.skill.meta_list("core-conflicts"))
            if any(
                c.skill.name in conflicts or cand.skill.name in set(c.skill.meta_list("core-conflicts"))
                for c in chosen
            ):
                continue
            if used + cand.skill.tokens > max_tokens:
                continue
            chosen.append(cand)
            used += cand.skill.tokens
        return chosen

    def catalog(self, *, exclude: set[str] | None = None, limit: int = 30) -> str:
        exclude = exclude or set()
        lines = []
        for skill in self.all():
            if skill.name in exclude or not skill.usable:
                continue
            lines.append(f"- {skill.name}: {skill.description[:220]}")
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def fingerprint(self) -> str:
        return sha256_hex("|".join(f"{s.name}:{s.sha256}:{s.trust}" for s in self.all()))

    def import_skills(
        self, source: str, dest_root: Path, *, subpath: str | None = None, names: list[str] | None = None
    ) -> list[Skill]:
        """Import skills from a local directory or git URL. Imported skills are untrusted."""
        with tempfile.TemporaryDirectory(prefix="core-skill-import-") as tmp:
            commit = None
            if re.match(r"^(https://|git@|ssh://)", source):
                proc = subprocess.run(
                    ["git", "clone", "--depth", "1", "--quiet", source, f"{tmp}/repo"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                if proc.returncode != 0:
                    raise CoreError(f"git clone failed: {proc.stderr.strip()[:300]}")
                base = Path(tmp) / "repo"
                commit = subprocess.run(
                    ["git", "-C", str(base), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
                ).stdout.strip()
            else:
                base = Path(source).expanduser().resolve()
                if not base.is_dir():
                    raise NotFoundError(f"skill source {source} is not a directory")
            search_root = base / subpath if subpath else base
            skill_dirs = (
                [search_root]
                if (search_root / "SKILL.md").exists()
                else [p.parent for p in sorted(search_root.rglob("SKILL.md"))]
            )
            imported: list[Skill] = []
            slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")[-60:] or "local"
            for d in skill_dirs:
                skill = load_skill(d, "imported")
                if names and skill.name not in names:
                    continue
                if not skill.valid:
                    continue
                target = dest_root / slug / skill.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(d, target, ignore=shutil.ignore_patterns(".git", "__pycache__"))
                (target / PROVENANCE_FILE).write_text(
                    json.dumps(
                        {
                            "source": source,
                            "subpath": str(d.relative_to(base)) if d != base else ".",
                            "commit": commit,
                            "imported_at": self.clock.now(),
                            "license": skill.frontmatter.get("license"),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                imported.append(load_skill(target, "imported"))
        self.refresh()
        return imported
