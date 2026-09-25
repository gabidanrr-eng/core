"""SQLite access with explicit transaction semantics.

* WAL journal, ``synchronous=NORMAL``, foreign keys on, generous busy timeout.
* Write transactions use ``BEGIN IMMEDIATE`` so writers serialize up-front instead of failing
  on lock upgrade; nested ``tx()`` calls become savepoints.
* ``after_commit`` callbacks run only after the outermost transaction commits (used to publish
  events to live subscribers without ever announcing state that was rolled back).
* One connection per thread. Code must never ``await`` inside ``tx()``: a guard raises if a
  second asyncio task tries to enter a transaction owned by another task on the same thread.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class _TxState:
    depth: int = 0
    after: list[Callable[[], None]] = field(default_factory=list)
    owner: int | None = None


def _current_task_id() -> int | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    return id(task) if task is not None else None


class Database:
    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 15000):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("Core Main requires a file-backed database (per-thread connections)")
        self.busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._lock = threading.Lock()
        self._conns: list[sqlite3.Connection] = []
        self._closed = False
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, timeout=self.busy_timeout_ms / 1000, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def conn(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("database is closed")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open()
            self._local.conn = conn
            with self._lock:
                self._conns.append(conn)
        return conn

    def _state(self) -> _TxState:
        state: _TxState | None = getattr(self._local, "tx", None)
        if state is None:
            state = _TxState()
            self._local.tx = state
        return state

    @property
    def in_transaction(self) -> bool:
        return self._state().depth > 0

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn()
        state = self._state()
        task_id = _current_task_id()
        if state.depth == 0:
            conn.execute("BEGIN IMMEDIATE")
            state.depth = 1
            state.owner = task_id
            try:
                yield conn
            except BaseException:
                state.depth = 0
                state.after.clear()
                state.owner = None
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            state.depth = 0
            state.owner = None
            try:
                conn.execute("COMMIT")
            except BaseException:
                state.after.clear()
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            callbacks, state.after = state.after, []
            for cb in callbacks:
                try:
                    cb()
                except Exception:  # an observer failure must never undo committed state
                    log.exception("after-commit callback failed")
            return
        if state.owner is not None and task_id is not None and task_id != state.owner:
            raise RuntimeError("transaction entered from a different asyncio task (awaiting inside tx?)")
        name = f"sp{state.depth}"
        mark = len(state.after)
        conn.execute(f"SAVEPOINT {name}")
        state.depth += 1
        try:
            yield conn
        except BaseException:
            conn.execute(f"ROLLBACK TO {name}")
            conn.execute(f"RELEASE {name}")
            del state.after[mark:]
            state.depth -= 1
            raise
        conn.execute(f"RELEASE {name}")
        state.depth -= 1

    def after_commit(self, callback: Callable[[], None]) -> None:
        state = self._state()
        if state.depth > 0:
            state.after.append(callback)
        else:
            callback()

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        return self.conn().execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn().execute(sql, params).fetchone()
        return row

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> Any:
        row = self.conn().execute(sql, params).fetchone()
        return None if row is None else row[0]

    def integrity_check(self, *, full: bool = False) -> list[str]:
        rows = self.query("PRAGMA integrity_check" if full else "PRAGMA quick_check")
        problems = [str(r[0]) for r in rows if str(r[0]) != "ok"]
        fk = self.query("PRAGMA foreign_key_check")
        problems.extend(f"foreign key violation in {r[0]} rowid={r[1]} → {r[2]}" for r in fk)
        return problems

    def backup_to(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(dest))
        try:
            self.conn().backup(target)
        finally:
            target.close()

    def checkpoint(self) -> None:
        self.conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, []
            self._closed = True
        for c in conns:
            try:
                c.close()
            except sqlite3.Error:
                pass
        self._local = threading.local()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if row is None else {k: row[k] for k in row.keys()}  # noqa: SIM118
