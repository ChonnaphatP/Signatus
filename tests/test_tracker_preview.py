from __future__ import annotations

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from signatus_ai.cache import DetectionCache
from signatus_ai.tracker import (
    APPROVED_MODEL_CLASS_NAMES,
    UltralyticsOpenVINOTracker,
    validate_model_class_names,
)
from signatus_contracts.frame_buffer import FrameDetection


class FakeCapture:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame


class FakeModel:
    def track(self, **_: object) -> list[SimpleNamespace]:
        return [SimpleNamespace(boxes=None)]


class RecordingPublisher:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.frame: object | None = None
        self.timestamp_ns: int | None = None
        self.detections: tuple[FrameDetection, ...] = ()

    def publish(
        self,
        frame: object,
        *,
        timestamp_ns: int | None = None,
        detections: tuple[FrameDetection, ...] = (),
    ) -> bool:
        if self.fail:
            raise RuntimeError("preview unavailable")
        self.frame = frame
        self.timestamp_ns = timestamp_ns
        self.detections = detections
        return True


async def unused_event_publish(_: object) -> None:
    raise AssertionError("empty snapshot must not publish a tracking event")


def tracker_with(publisher: RecordingPublisher) -> UltralyticsOpenVINOTracker:
    tracker = UltralyticsOpenVINOTracker(
        model_path="unused",
        camera_source=0,
        person_class="Person",
        lost_timeout_seconds=1.5,
        cache=DetectionCache(),
        publish=unused_event_publish,  # type: ignore[arg-type]
        frame_publisher=publisher,
    )
    tracker._capture = FakeCapture(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    tracker._model = FakeModel()
    return tracker


class TrackerPreviewTests(unittest.TestCase):
    def test_requires_exact_approved_model_class_ids_and_names(self) -> None:
        validate_model_class_names(APPROVED_MODEL_CLASS_NAMES)
        swapped = dict(APPROVED_MODEL_CLASS_NAMES)
        swapped[7], swapped[0] = swapped[0], swapped[7]

        with self.assertRaisesRegex(ValueError, "approved Signatus v1 class policy"):
            validate_model_class_names(swapped)

    def test_tracker_open_validates_runtime_model_names_before_opening_camera(self) -> None:
        tracker = UltralyticsOpenVINOTracker(
            model_path="model",
            camera_source=0,
            person_class="Person",
            lost_timeout_seconds=1.5,
            cache=DetectionCache(),
            publish=unused_event_publish,  # type: ignore[arg-type]
        )
        names = dict(APPROVED_MODEL_CLASS_NAMES)
        names[9] = "gloves"
        fake_ultralytics = ModuleType("ultralytics")
        fake_ultralytics.YOLO = lambda _path: SimpleNamespace(names=names)  # type: ignore[attr-defined]
        fake_cv2 = ModuleType("cv2")
        fake_cv2.VideoCapture = lambda _source: self.fail("camera opened too early")  # type: ignore[attr-defined]

        with (
            patch.dict("sys.modules", {"cv2": fake_cv2, "ultralytics": fake_ultralytics}),
            self.assertRaisesRegex(ValueError, "approved Signatus v1 class policy"),
        ):
            tracker._open()

    def test_raw_capture_frame_is_published_before_empty_detection_snapshot(self) -> None:
        publisher = RecordingPublisher()
        tracker = tracker_with(publisher)

        snapshot = tracker._read_snapshot()

        self.assertEqual(snapshot.people, ())
        self.assertEqual(snapshot.detections, ())
        self.assertIs(publisher.frame, tracker._capture.frame)
        self.assertIsNotNone(publisher.timestamp_ns)
        self.assertGreater(publisher.timestamp_ns or 0, 0)
        self.assertEqual(publisher.detections, ())

    def test_preview_exception_does_not_interrupt_tracking_snapshot(self) -> None:
        publisher = RecordingPublisher(fail=True)
        tracker = tracker_with(publisher)

        with self.assertLogs("signatus_ai.tracker", level="ERROR"):
            snapshot = tracker._read_snapshot()

        self.assertEqual(snapshot.people, ())
        self.assertEqual(snapshot.detections, ())


if __name__ == "__main__":
    unittest.main()
