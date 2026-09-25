"""Cooperative cancellation that propagates across layers.

A ``CancelToken`` is created per task attempt; child tokens are derived for model calls,
tool executions, subprocesses and browser sessions. Cancelling a parent cancels every child.
The durable side of cancellation (``tasks.cancel_requested_at``) is owned by TaskService; the
runtime fires the in-process token when it observes or issues a durable cancel.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable
from typing import TypeVar

from coremain.errors import OperationCancelled

T = TypeVar("T")


class CancelToken:
    def __init__(self, parent: CancelToken | None = None):
        self._event = asyncio.Event()
        self.reason: str | None = None
        self._children: list[CancelToken] = []
        if parent is not None:
            parent._children.append(self)
            if parent.cancelled:
                self.cancel(parent.reason or "parent cancelled")

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "cancelled") -> None:
        if self._event.is_set():
            return
        self.reason = reason
        self._event.set()
        for child in self._children:
            child.cancel(reason)

    def child(self) -> CancelToken:
        return CancelToken(self)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise OperationCancelled(self.reason or "cancelled")

    async def wait(self) -> None:
        await self._event.wait()

    async def run(self, awaitable: Awaitable[T]) -> T:
        """Await ``awaitable`` but abort (cancelling it) as soon as this token is cancelled."""
        self.raise_if_cancelled()
        task = asyncio.ensure_future(awaitable)
        waiter = asyncio.ensure_future(self._event.wait())
        try:
            done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            task.cancel()
            waiter.cancel()
            raise
        if task in done:
            waiter.cancel()
            return task.result()
        task.cancel()
        # The abandoned operation's own outcome is irrelevant once cancellation won.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        raise OperationCancelled(self.reason or "cancelled")

    async def sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._event.wait(), timeout=seconds)
        except TimeoutError:
            return
        raise OperationCancelled(self.reason or "cancelled")
