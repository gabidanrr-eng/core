"""Server-Sent Events parsing for streaming provider APIs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class SSEEvent:
    event: str
    data: str
    id: str | None = None


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    event = ""
    data: list[str] = []
    last_id: str | None = None
    async for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data or event:
                yield SSEEvent(event or "message", "\n".join(data), last_id)
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
        elif field == "id":
            last_id = value
    if data or event:
        yield SSEEvent(event or "message", "\n".join(data), last_id)
