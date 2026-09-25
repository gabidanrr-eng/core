"""Benchmark scenarios: a repository fixture, a request, and objective post-conditions.

Scenarios are data (YAML). Built-in families live in ``coremain/evals/data``; projects can add
their own in ``.core/evals/*.yaml`` and users in ``<config>/evals/*.yaml``. Resolved failures
promoted with ``core learn regress`` become scenarios in the ``regression`` suite.

A scenario may carry a ``script`` for the deterministic ``scripted`` provider. Scripted runs
(the ``smoke`` suite) validate Core Main's own machinery — routing, tools, workspaces,
verification, review and the evidence gate — without credentials or network. Live runs use the
configured models and ignore scripts. Script items are either full scripted-provider turns or a
compact ``{tool_name: {args}}`` mapping (a list of such mappings is one turn with parallel calls).
"""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from coremain.errors import ConfigError

DATA_DIR = Path(__file__).parent / "data"
TURN_KEYS = frozenset({"text", "tool_calls", "error", "malformed_arguments", "delay_s", "usage"})
CHECK_KINDS = frozenset(
    {
        "status",
        "file_contains",
        "file_not_contains",
        "file_exists",
        "file_absent",
        "file_matches",
        "command",
        "evidence",
        "min_level",
        "canonical_unchanged",
        "answer_contains",
        "review_verdict",
        "findings_min",
        "changed_only",
        "max_model_calls",
        "model_retries_min",
        "repo_not_contains",
    }
)
REQUIREMENTS = frozenset({"python", "git", "bash", "node", "browser"})


@dataclass
class Scenario:
    id: str
    family: str
    title: str
    request: str
    files: dict[str, str]
    checks: list[dict[str, Any]]
    suites: list[str] = field(default_factory=lambda: ["smoke", "core"])
    script: dict[str, Any] | None = None
    mode: str | None = None
    contract: dict[str, Any] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)
    test_command: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    uncommitted: dict[str, str] = field(default_factory=dict)
    models: int = 1
    timeout_s: float = 600.0
    source: str = "builtin"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["files"] = sorted(self.files)
        return data

    def expanded_test_command(self) -> str | None:
        return expand(self.test_command) if self.test_command else None

    def scripted_turns(self) -> dict[str, Any] | None:
        if not self.script:
            return None
        roles = {role: [_turn(item, self.id) for item in items] for role, items in self.script.items()}
        return {"name": f"eval:{self.id}", "roles": roles}

    def missing_requirements(self) -> list[str]:
        missing = []
        for req in self.requires:
            if req in {"git", "bash", "node"} and shutil.which(req) is None:
                missing.append(req)
            elif req == "browser" and importlib.util.find_spec("playwright") is None:
                missing.append("browser (playwright)")
        return missing


def expand(text: str) -> str:
    return text.replace("{python}", shlex.quote(sys.executable))


def _turn(item: Any, scenario_id: str) -> dict[str, Any]:
    if isinstance(item, list):
        calls = []
        for call in item:
            calls.extend(_turn(call, scenario_id)["tool_calls"])
        return {"tool_calls": calls}
    if not isinstance(item, dict) or not item:
        raise ConfigError(f"scenario {scenario_id}: script turns must be mappings, got {item!r}")
    if set(item) & TURN_KEYS:
        return dict(item)
    if len(item) != 1:
        raise ConfigError(
            f"scenario {scenario_id}: compact turn must have exactly one tool, got {sorted(item)}"
        )
    name, args = next(iter(item.items()))
    return {"tool_calls": [{"name": name, "arguments": args or {}}]}


def _parse(raw: dict[str, Any], origin: str) -> Scenario:
    missing = [k for k in ("id", "family", "title", "request", "checks") if k not in raw]
    if missing:
        raise ConfigError(f"{origin}: scenario is missing {', '.join(missing)}")
    unknown = set(raw) - {f for f in Scenario.__dataclass_fields__ if f != "source"}
    if unknown:
        raise ConfigError(f"{origin}: scenario {raw['id']} has unknown keys {sorted(unknown)}")
    for check in raw["checks"]:
        if not isinstance(check, dict) or len(check) != 1 or next(iter(check)) not in CHECK_KINDS:
            raise ConfigError(f"{origin}: scenario {raw['id']} has an invalid check {check!r}")
    for req in raw.get("requires", []):
        if req not in REQUIREMENTS:
            raise ConfigError(f"{origin}: scenario {raw['id']} requires unknown '{req}'")
    scenario = Scenario(**{k: v for k, v in raw.items()}, source=origin)
    scenario.files = {str(k): str(v) for k, v in (raw.get("files") or {}).items()}
    scenario.uncommitted = {str(k): str(v) for k, v in (raw.get("uncommitted") or {}).items()}
    scenario.scripted_turns()  # validate script shape early
    return scenario


def load_file(path: Path, origin: str | None = None) -> list[Scenario]:
    try:
        docs = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read scenarios from {path}: {exc}") from exc
    if isinstance(docs, dict):
        docs = docs.get("scenarios", [])
    if not isinstance(docs, list):
        raise ConfigError(f"{path}: expected a list of scenarios")
    return [_parse(d, origin or path.name) for d in docs]


def load_scenarios(extra_dirs: list[Path] | None = None) -> list[Scenario]:
    scenarios: dict[str, Scenario] = {}
    sources: list[tuple[Path, str]] = [(p, f"builtin:{p.name}") for p in sorted(DATA_DIR.glob("*.yaml"))]
    for d in extra_dirs or []:
        sources += [(p, str(p)) for p in sorted(d.glob("*.yaml"))]
    for path, origin in sources:
        for sc in load_file(path, origin):
            if sc.id in scenarios and not origin.startswith("builtin:"):
                scenarios[sc.id] = sc  # project/user definitions override built-ins deliberately
            elif sc.id in scenarios:
                raise ConfigError(f"duplicate scenario id {sc.id} in {origin}")
            else:
                scenarios[sc.id] = sc
    return sorted(scenarios.values(), key=lambda s: (s.family, s.id))


def regression_scenario(row: dict[str, Any]) -> Scenario:
    """Build a scenario from a ``regression_cases`` row (see coremain.learning.analysis)."""
    spec = dict(row["scenario"])
    spec.setdefault("suites", ["regression"])
    return _parse(spec, f"regression:{row['id']}")
