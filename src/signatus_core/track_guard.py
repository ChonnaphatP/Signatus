from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrackRecord:
    last_seen_at: float
    handled: bool = False
    face_failures: int = 0
    retry_not_before: float = 0.0


@dataclass(frozen=True, slots=True)
class FaceFailureDecision:
    attempt: int
    retry_allowed: bool


class InMemoryTrackGuard:
    """Process-local track lifecycle state. Nothing is written to disk."""

    def __init__(self, retry_cooldown_seconds: float = 1.0, max_face_failures: int = 3):
        if retry_cooldown_seconds < 0:
            raise ValueError("retry cooldown must be non-negative")
        if max_face_failures < 1:
            raise ValueError("max face failures must be at least one")
        self._retry_cooldown_seconds = retry_cooldown_seconds
        self._max_face_failures = max_face_failures
        self._tracks: dict[int, TrackRecord] = {}

    def observe(self, track_id: int, now: float) -> None:
        record = self._tracks.get(track_id)
        if record is None:
            self._tracks[track_id] = TrackRecord(last_seen_at=now)
        else:
            record.last_seen_at = now

    def should_trigger(self, track_id: int, now: float) -> bool:
        record = self._tracks.get(track_id)
        if record is None or record.handled:
            return False
        return now >= record.retry_not_before

    def mark_handled(self, track_id: int) -> None:
        record = self._tracks.get(track_id)
        if record is not None:
            record.handled = True

    def record_face_failure(self, track_id: int, now: float) -> FaceFailureDecision:
        record = self._tracks[track_id]
        record.face_failures += 1
        retry_allowed = record.face_failures < self._max_face_failures
        if retry_allowed:
            record.retry_not_before = now + self._retry_cooldown_seconds
        else:
            record.handled = True
        return FaceFailureDecision(record.face_failures, retry_allowed)

    def forget(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(self._tracks)
