from __future__ import annotations

import unittest

from signatus_ai.association import (
    AssociationUnresolvedError,
    SinglePersonFrameAssociation,
)
from signatus_ai.cache import BoundingBox, Detection, FrameSnapshot, TrackedPerson

BOX = BoundingBox(0.0, 0.0, 10.0, 10.0)


def snapshot(*track_ids: int) -> FrameSnapshot:
    return FrameSnapshot(
        captured_at=1000.0,
        people=tuple(TrackedPerson(track_id, 0.9, BOX) for track_id in track_ids),
        detections=(
            Detection("Person", 0.9, BOX),
            Detection("helmet", 0.8, BOX),
            Detection("none", 0.7, BOX),
        ),
    )


class SinglePersonFrameAssociationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.association = SinglePersonFrameAssociation(frozenset({"Person"}))

    def test_returns_non_person_classes_for_exactly_requested_track(self) -> None:
        classes = self.association.classes_for_track(7, snapshot(7))

        self.assertEqual(classes, ("helmet", "none"))

    def test_no_person_frame_is_unresolved(self) -> None:
        with self.assertRaises(AssociationUnresolvedError):
            self.association.classes_for_track(7, snapshot())

    def test_multiple_person_frame_is_unresolved(self) -> None:
        with self.assertRaises(AssociationUnresolvedError):
            self.association.classes_for_track(7, snapshot(7, 8))

    def test_different_single_track_is_unresolved(self) -> None:
        with self.assertRaises(AssociationUnresolvedError):
            self.association.classes_for_track(7, snapshot(8))


if __name__ == "__main__":
    unittest.main()
