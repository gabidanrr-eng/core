"""Stage strategies. Weights and caps are data, versioned so evaluations can attribute changes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Strategy:
    name: str
    version: str
    weights: dict[str, float]
    caps: dict[str, float] = field(default_factory=dict)  # max fraction of the budget per kind
    expand_depth: int = 1
    full_file_tokens: int = 3500
    search_hits: int = 12


_BASE = {
    "task": 10.0,
    "contract": 10.0,
    "instructions": 6.0,
    "profile": 5.0,
    "skill": 4.5,
    "skill_catalog": 2.0,
    "memory": 3.5,
    "decision": 3.5,
    "plan": 7.0,
    "prior": 6.0,
    "evidence": 5.0,
    "finding": 6.5,
    "failure": 5.0,
    "hypothesis": 4.0,
    "repo_map": 3.0,
    "file": 4.0,
    "outline": 2.5,
    "snippet": 3.0,
    "diff": 6.0,
    "research": 3.0,
    "git": 1.5,
    "conversation": 4.0,
}


def _w(**overrides: float) -> dict[str, float]:
    return {**_BASE, **overrides}


STRATEGIES: dict[str, Strategy] = {
    "question": Strategy(
        "question",
        "1",
        _w(file=4.5, snippet=4.0, repo_map=4.0, diff=1.0),
        {"file": 0.55, "snippet": 0.3},
        1,
        3000,
        16,
    ),
    "architecture": Strategy(
        "architecture",
        "1",
        _w(repo_map=6.0, decision=5.0, outline=4.0, file=3.0),
        {"outline": 0.35},
        2,
        2000,
        10,
    ),
    "planning": Strategy(
        "planning", "1", _w(repo_map=5.0, outline=3.5, decision=4.5, memory=4.0), {"file": 0.45}, 1, 3000, 12
    ),
    "implementation": Strategy(
        "implementation", "1", _w(file=5.5, snippet=3.5, plan=8.0), {"file": 0.6}, 1, 5000, 10
    ),
    "debugging": Strategy(
        "debugging",
        "1",
        _w(evidence=8.0, failure=7.0, hypothesis=6.0, file=5.0, diff=5.0),
        {"evidence": 0.25},
        1,
        4500,
        10,
    ),
    "testing": Strategy("testing", "1", _w(file=5.0, outline=3.5, evidence=6.0), {"file": 0.55}, 1, 4000, 10),
    "review": Strategy(
        "review",
        "1",
        _w(diff=9.0, evidence=7.0, finding=6.0, file=3.5, plan=5.0),
        {"diff": 0.45, "file": 0.3},
        1,
        3000,
        6,
    ),
    "security": Strategy("security", "1", _w(diff=9.0, file=4.0, evidence=6.0), {"diff": 0.5}, 1, 3000, 8),
    "browser": Strategy("browser", "1", _w(file=4.5, evidence=7.0, snippet=3.0), {"file": 0.5}, 1, 3500, 8),
    "documentation": Strategy(
        "documentation", "1", _w(repo_map=5.5, outline=4.5, file=3.5), {"outline": 0.4}, 1, 3000, 10
    ),
    "correction": Strategy(
        "correction",
        "1",
        _w(finding=9.0, evidence=8.5, diff=7.0, file=5.0, plan=6.0),
        {"diff": 0.35},
        1,
        4500,
        6,
    ),
}


def strategy_for(stage: str) -> Strategy:
    return STRATEGIES.get(stage, STRATEGIES["implementation"])
