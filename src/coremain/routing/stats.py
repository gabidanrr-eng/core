"""Measured model performance from durable call records (feeds routing)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from coremain.store.db import Database


@dataclass
class ModelStat:
    calls: int = 0
    errors: int = 0
    malformed: int = 0
    blocking_errors: int = 0
    last_blocking_class: str | None = None
    last_blocking_at: float | None = None
    p50_latency_ms: float | None = None

    @property
    def reliability(self) -> float:
        # Beta(4,1) prior: an unmeasured model starts at 0.8 and converges to its record.
        good = self.calls - self.errors - self.malformed
        return (good + 4) / (self.calls + 5)


class ModelStats:
    def __init__(self, db: Database, *, ttl_s: float = 30.0, window_s: float = 30 * 86400):
        self.db = db
        self.ttl_s = ttl_s
        self.window_s = window_s
        self._cache: dict[tuple[str, str | None], tuple[float, ModelStat]] = {}

    def get(self, model_key: str, role: str | None = None) -> ModelStat:
        key = (model_key, role)
        hit = self._cache.get(key)
        now = time.time()
        if hit and now - hit[0] < self.ttl_s:
            return hit[1]
        params: list[object] = [model_key, now - self.window_s]
        role_sql = ""
        if role:
            role_sql = " AND role = ?"
            params.append(role)
        row = self.db.one(
            "SELECT COUNT(*) AS calls, SUM(status = 'error') AS errors, SUM(error_class = 'malformed_tool_arguments') AS malformed "
            f"FROM model_calls WHERE model_key = ? AND started_at > ?{role_sql} AND status IN ('ok','error')",
            params,
        )
        stat = ModelStat(int(row["calls"] or 0), int(row["errors"] or 0), int(row["malformed"] or 0)) if row else ModelStat()
        blocking = self.db.one(
            "SELECT error_class, started_at FROM model_calls WHERE model_key = ? AND error_class IN "
            "('auth_failed','insufficient_funds','model_unavailable') ORDER BY started_at DESC LIMIT 1",
            (model_key,),
        )
        if blocking is not None:
            later_ok = self.db.scalar("SELECT COUNT(*) FROM model_calls WHERE model_key = ? AND status = 'ok' AND started_at > ?",
                                      (model_key, blocking["started_at"]))
            if not later_ok:
                stat.last_blocking_class = blocking["error_class"]
                stat.last_blocking_at = blocking["started_at"]
        lat = self.db.query(
            "SELECT latency_ms FROM model_calls WHERE model_key = ? AND status = 'ok' AND latency_ms IS NOT NULL "
            "ORDER BY started_at DESC LIMIT 50", (model_key,),
        )
        if lat:
            values = sorted(r["latency_ms"] for r in lat)
            stat.p50_latency_ms = float(values[len(values) // 2])
        self._cache[key] = (now, stat)
        return stat

    def invalidate(self) -> None:
        self._cache.clear()
