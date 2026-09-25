"""Capability-aware router.

Model choice is driven by the problem: hard requirements (tool calling, context size, vision,
role restrictions, availability) filter candidates, then a transparent score combines the
role-relevant declared strengths, measured reliability, cost and latency according to the
user's preference and the task's risk. Every decision records concise reasons. There is no
fixed cheap/medium/expensive ladder and no model is ever used unless the user configured it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coremain.config.schema import RoutingConfig
from coremain.providers.errors import ProviderError, ProviderErrorClass
from coremain.providers.registry import ModelProfile, ProviderRegistry
from coremain.routing.stats import ModelStats

ROLE_DIMENSIONS: dict[str, dict[str, float]] = {
    "planner": {"planning": 1.0, "coding": 0.4, "long_context": 0.3},
    "implementer": {"coding": 1.0, "tool_use": 0.6, "debugging": 0.3},
    "researcher": {"research": 1.0, "long_context": 0.5, "writing": 0.3},
    "reviewer": {"review": 1.0, "security": 0.4, "coding": 0.4},
    "debugger": {"debugging": 1.0, "coding": 0.6, "tool_use": 0.4},
    "verifier": {"review": 0.6, "coding": 0.5},
    "summarizer": {"writing": 1.0},
}
PREFERENCE_WEIGHTS = {
    "quality": {"quality": 1.0, "reliability": 0.5, "cost": 0.05, "latency": 0.05},
    "balanced": {"quality": 0.8, "reliability": 0.5, "cost": 0.25, "latency": 0.15},
    "speed": {"quality": 0.5, "reliability": 0.4, "cost": 0.1, "latency": 0.6},
    "cost": {"quality": 0.5, "reliability": 0.4, "cost": 0.7, "latency": 0.1},
}
LATENCY_SCORE = {"fast": 1.0, "standard": 0.6, "slow": 0.25}
BLOCKED_FOR_S = 900.0


@dataclass
class RouteRequirements:
    role: str
    task_kind: str = "general"
    needs_tools: bool = True
    needs_vision: bool = False
    min_context: int = 0
    risk: str = "normal"  # low | normal | high
    latency_sensitive: bool = False
    extra_dimensions: dict[str, float] = field(default_factory=dict)
    avoid_models: set[str] = field(default_factory=set)
    independent_of: str | None = None
    pin: str | None = None


@dataclass
class RouteDecision:
    role: str
    model: ModelProfile
    score: float
    reasons: list[str]
    candidates: list[dict[str, Any]]
    independence: str | None = None
    pinned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "model": self.model.ref, "model_key": self.model.key, "score": round(self.score, 3),
                "reasons": self.reasons, "independence": self.independence, "pinned": self.pinned, "candidates": self.candidates}


class Router:
    def __init__(self, registry: ProviderRegistry, config: RoutingConfig, stats: ModelStats, *,
                 heuristics: list[dict[str, Any]] | None = None, clock_now: Any = None):
        self.registry = registry
        self.config = config
        self.stats = stats
        self.heuristics = heuristics or []
        self._now = clock_now

    def route(self, req: RouteRequirements) -> RouteDecision:
        models = self.registry.models()
        if not models:
            raise ProviderError("no models are configured", error_class=ProviderErrorClass.NOT_CONFIGURED,
                                hint="Add a provider and model: `core providers add openai` then `core models add ...`.")
        pin = req.pin or self.config.pins.get(req.role)
        if pin:
            profile = self.registry.model(pin)
            problems = self._hard_problems(profile, req)
            if problems:
                raise ProviderError(f"pinned model {profile.ref} cannot serve role {req.role}: {problems}",
                                    error_class=ProviderErrorClass.INVALID_REQUEST, provider_id=profile.provider_id,
                                    model_id=profile.model_id)
            return RouteDecision(req.role, profile, 1.0, [f"pinned for role '{req.role}'" if not req.pin else "pinned for this task"],
                                 [{"model": profile.ref, "pinned": True}], self._independence(profile, req), pinned=True)
        allowed = set(self.config.allowed) if self.config.allowed else None
        prefer = self.config.prefer
        weights = dict(PREFERENCE_WEIGHTS[prefer])
        if req.risk == "high":
            weights["quality"] *= 1.3
            weights["cost"] *= 0.5
        elif req.risk == "low":
            weights["cost"] *= 1.3
        if req.latency_sensitive:
            weights["latency"] *= 2.0
        dims = dict(ROLE_DIMENSIONS.get(req.role, {"coding": 1.0}))
        for dim, w in req.extra_dimensions.items():
            dims[dim] = dims.get(dim, 0.0) + w
        costs = [self._cost_index(m) for m in models]
        known_costs = [c for c in costs if c is not None]
        max_cost = max(known_costs) if known_costs else None
        scored: list[tuple[float, ModelProfile, list[str]]] = []
        report: list[dict[str, Any]] = []
        for m in models:
            if allowed is not None and m.key not in allowed:
                report.append({"model": m.ref, "rejected": "not in routing.allowed"})
                continue
            problems = self._hard_problems(m, req)
            if problems:
                report.append({"model": m.ref, "rejected": problems})
                continue
            reasons: list[str] = []
            total_w = sum(dims.values()) or 1.0
            quality = sum(m.strength(d) * w for d, w in dims.items()) / total_w
            top_dim = max(dims, key=lambda d: dims[d])
            reasons.append(f"{top_dim} strength {m.strength(top_dim):.2f}" + (" (declared)" if top_dim in m.config.strengths else f" ({m.config.tier} tier prior)"))
            stat = self.stats.get(m.key)
            reliability = stat.reliability
            if stat.calls:
                reasons.append(f"measured reliability {reliability:.0%} over {stat.calls} calls")
            cost_index = self._cost_index(m)
            cost_score = 0.5 if cost_index is None or not max_cost else 1.0 - (cost_index / max_cost)
            if cost_index is not None and max_cost and cost_index == min(known_costs):
                reasons.append("lowest declared cost")
            latency_score = LATENCY_SCORE[m.config.latency]
            if stat.p50_latency_ms is not None:
                latency_score = 0.5 * latency_score + 0.5 * max(0.0, 1.0 - stat.p50_latency_ms / 60_000)
            score = (weights["quality"] * quality + weights["reliability"] * reliability + weights["cost"] * cost_score
                     + weights["latency"] * latency_score)
            if req.independent_of:
                if m.key == req.independent_of:
                    score -= 0.35 if self.config.independent_review != "off" else 0.0
                else:
                    implementer = self._safe_model(req.independent_of)
                    same_provider = implementer is not None and implementer.provider_id == m.provider_id
                    score += 0.08 if not same_provider else 0.04
                    reasons.append("independent of the implementer model" + (" and provider" if not same_provider else ""))
            for h in self.heuristics:
                if h.get("model") in (m.key, m.ref) and h.get("task_kind") in (None, req.task_kind) and h.get("role") in (None, req.role):
                    adj = float(h.get("adjust", 0.0))
                    score += adj
                    reasons.append(f"learned heuristic {h.get('name')} ({adj:+.2f})")
            if req.min_context > 0.6 * m.config.context_window:
                score -= 0.05
                reasons.append("context close to window limit")
            scored.append((score, m, reasons))
            report.append({"model": m.ref, "score": round(score, 3), "quality": round(quality, 3), "reliability": round(reliability, 3),
                           "cost": round(cost_score, 3), "latency": round(latency_score, 3)})
        if not scored:
            raise ProviderError(
                f"no configured model satisfies the requirements for role '{req.role}'",
                error_class=ProviderErrorClass.MODEL_UNAVAILABLE,
                hint="; ".join(f"{r['model']}: {r.get('rejected')}" for r in report if r.get("rejected")) or None,
            )
        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best, reasons = scored[0]
        reasons = [f"prefer={prefer}", *([f"risk={req.risk}"] if req.risk != "normal" else []), *reasons]
        return RouteDecision(req.role, best, best_score, reasons, report, self._independence(best, req))

    def _safe_model(self, key: str) -> ModelProfile | None:
        try:
            return self.registry.model(key)
        except Exception:  # noqa: BLE001
            return None

    def _independence(self, chosen: ModelProfile, req: RouteRequirements) -> str | None:
        if not req.independent_of:
            return None
        if chosen.key == req.independent_of:
            return "same model as implementer, fresh independent context"
        other = self._safe_model(req.independent_of)
        if other is not None and other.provider_id == chosen.provider_id:
            return "different model, same provider"
        return "different model and provider"

    def _hard_problems(self, m: ModelProfile, req: RouteRequirements) -> str | None:
        caps = m.config.capabilities
        if m.key in req.avoid_models:
            return "excluded for this request"
        if m.config.roles and req.role not in m.config.roles:
            return f"restricted to roles {m.config.roles}"
        if req.needs_tools and not caps.tool_calling:
            return "no tool calling"
        if req.needs_vision and not caps.vision:
            return "no vision"
        if req.min_context and req.min_context > m.config.context_window:
            return f"context window {m.config.context_window} < required {req.min_context}"
        stat = self.stats.get(m.key)
        if stat.last_blocking_class and stat.last_blocking_at is not None:
            import time as _time

            now = self._now() if callable(self._now) else _time.time()
            if now - stat.last_blocking_at < BLOCKED_FOR_S:
                return f"recently blocked ({stat.last_blocking_class}); fix and retry or wait"
        return None

    @staticmethod
    def _cost_index(m: ModelProfile) -> float | None:
        c = m.config.cost
        if c.input_per_mtok is None and c.output_per_mtok is None:
            return None
        return (c.input_per_mtok or 0.0) * 0.75 + (c.output_per_mtok or 0.0) * 0.25
