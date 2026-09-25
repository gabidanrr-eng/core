"""Data-driven workflows: explicit nodes, outcomes and edges (inspectable, resumable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    role: str | None = None
    stage: str | None = None
    next: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    max_visits: int = 1


@dataclass(frozen=True)
class Workflow:
    name: str
    version: str
    start: str
    nodes: dict[str, Node]

    def describe(self) -> list[dict[str, Any]]:
        return [{"id": n.id, "kind": n.kind, "role": n.role, "next": n.next} for n in self.nodes.values()]


def _tail(repairs: int) -> dict[str, Node]:
    return {
        "verify": Node("verify", "verify", next={"ok": "review", "fail": "correct"}, max_visits=repairs + 2),
        "review": Node("review", "review", role="reviewer", stage="review",
                       next={"approve": "finalize", "request_changes": "correct", "inconclusive": "finalize"}, max_visits=repairs + 2),
        "correct": Node("correct", "implement", role="corrector", stage="correction", params={"correction": True},
                        next={"ok": "verify", "no_changes": "verify", "fail": "finalize", "exhausted": "finalize"}, max_visits=repairs),
        "finalize": Node("finalize", "finalize"),
    }


def build_workflows(repairs: int = 2) -> dict[str, Workflow]:
    tail = _tail(repairs)
    impl_next = {"ok": "verify", "no_changes": "finalize", "fail": "finalize"}
    return {
        "answer": Workflow("answer", "1", "research", {
            "research": Node("research", "answer", role="researcher", stage="question", next={"ok": "finalize", "fail": "finalize"}),
            "finalize": Node("finalize", "finalize"),
        }),
        "review": Workflow("review", "1", "review_existing", {
            "review_existing": Node("review_existing", "review_existing", role="reviewer", stage="review",
                                    next={"ok": "finalize", "fail": "finalize"}),
            "finalize": Node("finalize", "finalize"),
        }),
        "direct": Workflow("direct", "1", "implement", {
            "implement": Node("implement", "implement", role="implementer", stage="implementation", next=impl_next),
            **tail,
        }),
        "plan": Workflow("plan", "1", "plan", {
            "plan": Node("plan", "plan", role="planner", stage="planning", next={"ok": "implement", "fail": "implement"}),
            "implement": Node("implement", "implement", role="implementer", stage="implementation", next=impl_next),
            **tail,
        }),
        "collaborative": Workflow("collaborative", "1", "propose", {
            "propose": Node("propose", "propose", role="planner", stage="architecture", next={"ok": "synthesize", "fail": "implement"}),
            "synthesize": Node("synthesize", "synthesize", role="planner", stage="architecture", next={"ok": "implement", "fail": "implement"}),
            "implement": Node("implement", "implement", role="implementer", stage="implementation", next=impl_next),
            **tail,
        }),
        "debug": Workflow("debug", "1", "reproduce", {
            "reproduce": Node("reproduce", "reproduce", next={"ok": "investigate", "skip": "investigate", "fail": "investigate"}),
            "investigate": Node("investigate", "implement", role="debugger", stage="debugging", next=impl_next),
            **tail,
        }),
    }
