"""Leases with monotonic fencing tokens.

Every active piece of work holds a lease on a named resource (``task:<id>``). Each acquisition
increments the resource's token. Mutations performed on behalf of a worker verify, inside the
same transaction, that the worker's token is still current and unexpired, so a stalled or
partitioned worker that lost its lease cannot write late results.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from coremain.errors import LeaseLostError
from coremain.store.db import Database
from coremain.util.clock import Clock


@dataclass(frozen=True)
class Fence:
    resource: str
    owner: str
    token: int


class LeaseManager:
    def __init__(self, db: Database, clock: Clock):
        self.db = db
        self.clock = clock

    def acquire(self, resource: str, owner: str, ttl_s: float) -> Fence | None:
        now = self.clock.now()
        with self.db.tx() as conn:
            row = conn.execute("SELECT owner, token, expires_at, released_at FROM leases WHERE resource = ?", (resource,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO leases(resource, owner, token, acquired_at, expires_at) VALUES (?,?,?,?,?)",
                    (resource, owner, 1, now, now + ttl_s),
                )
                return Fence(resource, owner, 1)
            held = row["owner"] is not None and row["released_at"] is None and (row["expires_at"] or 0) > now
            if held and row["owner"] != owner:
                return None
            token = int(row["token"]) + 1
            conn.execute(
                "UPDATE leases SET owner = ?, token = ?, acquired_at = ?, expires_at = ?, released_at = NULL WHERE resource = ?",
                (owner, token, now, now + ttl_s, resource),
            )
            return Fence(resource, owner, token)

    def renew(self, fence: Fence, ttl_s: float) -> bool:
        now = self.clock.now()
        with self.db.tx() as conn:
            cur = conn.execute(
                "UPDATE leases SET expires_at = ? WHERE resource = ? AND owner = ? AND token = ? "
                "AND released_at IS NULL AND expires_at > ?",
                (now + ttl_s, fence.resource, fence.owner, fence.token, now),
            )
            return cur.rowcount == 1

    def release(self, fence: Fence) -> bool:
        with self.db.tx() as conn:
            cur = conn.execute(
                "UPDATE leases SET owner = NULL, released_at = ? WHERE resource = ? AND token = ? AND owner = ?",
                (self.clock.now(), fence.resource, fence.token, fence.owner),
            )
            return cur.rowcount == 1

    def verify(self, conn: sqlite3.Connection, fence: Fence) -> None:
        row = conn.execute("SELECT owner, token, expires_at, released_at FROM leases WHERE resource = ?", (fence.resource,)).fetchone()
        if row is None or int(row["token"]) != fence.token or row["owner"] != fence.owner or row["released_at"] is not None:
            raise LeaseLostError(
                f"lease on {fence.resource} is no longer held by this worker (token {fence.token})",
                details={"resource": fence.resource, "token": fence.token},
            )
        if (row["expires_at"] or 0) <= self.clock.now():
            raise LeaseLostError(f"lease on {fence.resource} expired", details={"resource": fence.resource})

    def expired(self) -> list[sqlite3.Row]:
        return self.db.query(
            "SELECT * FROM leases WHERE owner IS NOT NULL AND released_at IS NULL AND expires_at <= ?",
            (self.clock.now(),),
        )

    def force_expire(self, resource: str) -> None:
        """Administrative/test helper: expire a lease immediately (the holder is fenced out)."""
        with self.db.tx() as conn:
            conn.execute("UPDATE leases SET expires_at = ? WHERE resource = ?", (self.clock.now() - 1, resource))

    def holder(self, resource: str) -> sqlite3.Row | None:
        return self.db.one("SELECT * FROM leases WHERE resource = ?", (resource,))
