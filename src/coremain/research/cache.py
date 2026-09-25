"""TTL cache for research results on the shared ``cache_entries`` table.

Values are JSON documents (already redacted by the caller). Expired or undecodable rows are deleted
on read, so a corrupted entry degrades to a cache miss instead of an error.
"""

from __future__ import annotations

import json
from typing import Any

from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.jsonutil import dumps

NAMESPACE_PREFIX = "research."


class ResearchCache:
    def __init__(self, db: Database, clock: Clock):
        self.db = db
        self.clock = clock

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        row = self.db.one(
            "SELECT value, expires_at FROM cache_entries WHERE namespace = ? AND key = ?", (namespace, key)
        )
        if row is None:
            return None
        if row["expires_at"] is not None and float(row["expires_at"]) <= self.clock.now():
            self._delete(namespace, key)
            return None
        try:
            raw = row["value"]
            value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        except (UnicodeDecodeError, ValueError, AttributeError):
            value = None
        if not isinstance(value, dict):
            self._delete(namespace, key)
            return None
        self.db.execute(
            "UPDATE cache_entries SET hits = hits + 1 WHERE namespace = ? AND key = ?", (namespace, key)
        )
        return value

    def put(self, namespace: str, key: str, value: dict[str, Any], ttl_s: float) -> None:
        if ttl_s <= 0:
            return
        blob = dumps(value).encode("utf-8")
        now = self.clock.now()
        self.db.execute(
            "INSERT INTO cache_entries(namespace, key, value, size, created_at, expires_at, hits) VALUES (?,?,?,?,?,?,0) "
            "ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value, size = excluded.size, "
            "created_at = excluded.created_at, expires_at = excluded.expires_at, hits = 0",
            (namespace, key, blob, len(blob), now, now + ttl_s),
        )
        self.db.execute(
            "DELETE FROM cache_entries WHERE namespace = ? AND expires_at IS NOT NULL AND expires_at <= ?",
            (namespace, now),
        )

    def stats(self) -> dict[str, dict[str, int]]:
        rows = self.db.query(
            "SELECT namespace, COUNT(*) AS entries, COALESCE(SUM(size), 0) AS bytes, COALESCE(SUM(hits), 0) AS hits "
            "FROM cache_entries WHERE namespace LIKE ? GROUP BY namespace ORDER BY namespace",
            (NAMESPACE_PREFIX + "%",),
        )
        return {
            r["namespace"]: {"entries": int(r["entries"]), "bytes": int(r["bytes"]), "hits": int(r["hits"])}
            for r in rows
        }

    def clear(self, namespace: str | None = None) -> int:
        if namespace is None:
            cur = self.db.execute(
                "DELETE FROM cache_entries WHERE namespace LIKE ?", (NAMESPACE_PREFIX + "%",)
            )
        else:
            cur = self.db.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
        return int(cur.rowcount)

    def _delete(self, namespace: str, key: str) -> None:
        self.db.execute("DELETE FROM cache_entries WHERE namespace = ? AND key = ?", (namespace, key))
