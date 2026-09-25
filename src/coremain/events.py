"""Unified execution events.

Durable events are written in the same transaction as the state change they describe and
published to in-process subscribers only after commit. Ephemeral events (streaming deltas,
progress ticks) go to live subscribers only. Other processes follow durable events by
tailing the ``events`` table by sequence number, so the database doubles as a cross-process
event log without a daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from coremain.security.redact import Redactor
from coremain.store.db import Database
from coremain.util.clock import Clock
from coremain.util.jsonutil import dumps, loads


@dataclass
class Event:
    kind: str
    ts: float
    seq: int | None = None
    level: str = "info"
    project_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    actor: str = "runtime"
    data: dict[str, Any] = field(default_factory=dict)
    durable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "kind": self.kind,
            "level": self.level,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "actor": self.actor,
            "data": self.data,
            "durable": self.durable,
        }


Predicate = Callable[[Event], bool]


class Subscription:
    def __init__(self, bus: EventBus, loop: asyncio.AbstractEventLoop, predicate: Predicate | None, maxsize: int):
        self._bus = bus
        self._loop = loop
        self._predicate = predicate
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self._closed = False

    def _put(self, event: Event) -> None:
        if self._closed:
            return
        if self._queue.full():
            # Live views must never block the runtime; drop the oldest event and count it.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self.dropped += 1
        self._queue.put_nowait(event)

    def offer(self, event: Event) -> None:
        if self._closed or (self._predicate is not None and not self._predicate(event)):
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._put(event)
        elif not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._put, event)

    async def get(self, timeout: float | None = None) -> Event | None:
        try:
            if timeout is None:
                return await self._queue.get()
            return await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return None

    def get_nowait(self) -> Event | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self) -> None:
        self._closed = True
        self._bus._remove(self)

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> Event:
        if self._closed:
            raise StopAsyncIteration
        return await self._queue.get()


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()

    def subscribe(self, predicate: Predicate | None = None, *, maxsize: int = 10_000) -> Subscription:
        sub = Subscription(self, asyncio.get_running_loop(), predicate, maxsize)
        with self._lock:
            self._subs.append(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def publish(self, event: Event) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub.offer(event)


class EventLog:
    """Records durable events and fans them out to live subscribers after commit."""

    def __init__(self, db: Database, bus: EventBus, redactor: Redactor, clock: Clock):
        self.db = db
        self.bus = bus
        self.redactor = redactor
        self.clock = clock

    def emit(
        self,
        kind: str,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        actor: str = "runtime",
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> Event:
        payload = self.redactor.redact_obj(data or {})
        ts = self.clock.now()
        with self.db.tx() as conn:
            cur = conn.execute(
                "INSERT INTO events(ts, kind, level, project_id, session_id, task_id, attempt_id, actor, data_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, kind, level, project_id, session_id, task_id, attempt_id, actor, dumps(payload)),
            )
            event = Event(kind, ts, cur.lastrowid, level, project_id, session_id, task_id, attempt_id, actor, payload)
            self.db.after_commit(lambda: self.bus.publish(event))
        return event

    def ephemeral(
        self,
        kind: str,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.bus.publish(
            Event(kind, self.clock.now(), None, "debug", project_id, session_id, task_id, attempt_id, "runtime",
                  self.redactor.redact_obj(data or {}), durable=False)
        )

    @staticmethod
    def _row(row: Any) -> Event:
        return Event(
            kind=row["kind"], ts=row["ts"], seq=row["seq"], level=row["level"], project_id=row["project_id"],
            session_id=row["session_id"], task_id=row["task_id"], attempt_id=row["attempt_id"],
            actor=row["actor"], data=loads(row["data_json"], {}),
        )

    def since(self, seq: int, *, task_id: str | None = None, session_id: str | None = None, limit: int = 1000) -> list[Event]:
        sql = "SELECT * FROM events WHERE seq > ?"
        params: list[Any] = [seq]
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY seq LIMIT ?"
        params.append(limit)
        return [self._row(r) for r in self.db.query(sql, params)]

    def for_task(self, task_id: str, *, kinds: tuple[str, ...] | None = None, limit: int = 5000) -> list[Event]:
        sql = "SELECT * FROM events WHERE task_id = ?"
        params: list[Any] = [task_id]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        sql += " ORDER BY seq LIMIT ?"
        params.append(limit)
        return [self._row(r) for r in self.db.query(sql, params)]

    def latest_seq(self) -> int:
        return int(self.db.scalar("SELECT COALESCE(MAX(seq), 0) FROM events") or 0)
