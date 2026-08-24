from __future__ import annotations

from typing import Protocol

from .cache import FrameSnapshot


class AssociationUnresolvedError(RuntimeError):
    pass


class PPEAssociation(Protocol):
    def classes_for_track(self, track_id: int, snapshot: FrameSnapshot) -> tuple[str, ...]: ...


class UnconfiguredAssociation:
    def classes_for_track(self, track_id: int, snapshot: FrameSnapshot) -> tuple[str, ...]:
        raise AssociationUnresolvedError("Person-to-PPE association is not configured")


class SinglePersonFrameAssociation:
    """Approved v1 adapter for a strictly one-person entrance lane.

    PPE detections are returned only when the cached frame contains exactly the
    requested tracked person. Zero-person, multi-person, and track-mismatch
    frames remain unresolved and therefore fail closed in Core.
    """

    def __init__(self, excluded_classes: frozenset[str]):
        self._excluded_classes = {value.casefold() for value in excluded_classes}

    def classes_for_track(self, track_id: int, snapshot: FrameSnapshot) -> tuple[str, ...]:
        if len(snapshot.people) != 1 or snapshot.people[0].track_id != track_id:
            raise AssociationUnresolvedError("Cached frame is not a single-person frame")
        return tuple(
            detection.class_name
            for detection in snapshot.detections
            if detection.class_name.casefold() not in self._excluded_classes
        )
