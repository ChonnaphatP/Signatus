from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    track_id: int
    confidence: float
    box: BoundingBox


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    captured_at: float
    people: tuple[TrackedPerson, ...]
    detections: tuple[Detection, ...]
    frame: object | None = None


class DetectionCache:
    def __init__(self) -> None:
        self._latest: FrameSnapshot | None = None
        self._lock = asyncio.Lock()

    async def store(self, snapshot: FrameSnapshot) -> None:
        async with self._lock:
            self._latest = snapshot

    async def latest(self) -> FrameSnapshot | None:
        async with self._lock:
            return self._latest

    async def clear(self) -> None:
        """Invalidate all camera-derived decision data."""

        async with self._lock:
            self._latest = None
