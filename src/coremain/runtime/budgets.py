"""Bounded execution: turns, tokens, cost and wall time per task (across attempts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coremain.config.schema import BudgetConfig
from coremain.errors import BudgetExceededError
from coremain.util.clock import Clock


@dataclass
class BudgetTracker:
    config: BudgetConfig
    clock: Clock
    started_at: float
    turns: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    model_calls: int = 0
    previous_wall_s: float = 0.0
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def resume(cls, config: BudgetConfig, clock: Clock, previous: list[dict[str, Any]]) -> BudgetTracker:
        tracker = cls(config, clock, clock.now())
        for usage in previous:
            tracker.turns += int(usage.get("turns", 0))
            tracker.tokens += int(usage.get("tokens", 0))
            tracker.cost_usd += float(usage.get("cost_usd", 0.0))
            tracker.model_calls += int(usage.get("model_calls", 0))
            tracker.previous_wall_s += float(usage.get("wall_s", 0.0))
        return tracker

    def record(self, model_ref: str, input_tokens: int, output_tokens: int, cost: float | None) -> None:
        self.turns += 1
        self.model_calls += 1
        self.tokens += input_tokens + output_tokens
        if cost:
            self.cost_usd += cost
        m = self.per_model.setdefault(model_ref, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["tokens"] += input_tokens + output_tokens
        m["cost_usd"] += cost or 0.0

    @property
    def wall_s(self) -> float:
        return self.previous_wall_s + (self.clock.now() - self.started_at)

    def check(self) -> None:
        c = self.config
        if self.turns >= c.max_turns_per_task:
            raise BudgetExceededError(f"turn budget exhausted ({self.turns}/{c.max_turns_per_task})", details={"budget": "turns"})
        if self.tokens >= c.max_tokens_per_task:
            raise BudgetExceededError(f"token budget exhausted ({self.tokens}/{c.max_tokens_per_task})", details={"budget": "tokens"})
        if c.max_cost_usd_per_task is not None and self.cost_usd >= c.max_cost_usd_per_task:
            raise BudgetExceededError(f"cost budget exhausted (${self.cost_usd:.4f}/${c.max_cost_usd_per_task})",
                                      details={"budget": "cost"})
        if self.wall_s >= c.max_wall_time_s:
            raise BudgetExceededError(f"wall-time budget exhausted ({self.wall_s:.0f}s/{c.max_wall_time_s}s)", details={"budget": "time"})

    def snapshot(self) -> dict[str, Any]:
        return {"turns": self.turns, "tokens": self.tokens, "cost_usd": round(self.cost_usd, 6), "model_calls": self.model_calls,
                "wall_s": round(self.clock.now() - self.started_at, 2), "per_model": self.per_model}
