"""Injectable clock so leases, expiry and timeouts are testable deterministically."""

from __future__ import annotations

import asyncio
import time


class Clock:
    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock(Clock):
    def __init__(self, start: float = 1_800_000_000.0):
        self._t = start
        self._mono = 0.0

    def now(self) -> float:
        return self._t

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._t += seconds
        self._mono += seconds

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        await asyncio.sleep(0)


SYSTEM_CLOCK = Clock()
