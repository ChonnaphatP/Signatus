from __future__ import annotations

import asyncio

from signatus_contracts import AIEvent


class EventHub:
    def __init__(self, queue_size: int = 64):
        self._subscribers: set[asyncio.Queue[AIEvent]] = set()
        self._queue_size = queue_size

    def subscribe(self) -> asyncio.Queue[AIEvent]:
        queue: asyncio.Queue[AIEvent] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[AIEvent]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: AIEvent) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)
