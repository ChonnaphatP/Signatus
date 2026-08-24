from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from signatus_contracts import AIEvent, AIEventType
from signatus_contracts.frame_buffer import FrameDetection

from .cache import BoundingBox, Detection, DetectionCache, FrameSnapshot, TrackedPerson
from .camera import CameraOperationError

logger = logging.getLogger(__name__)

APPROVED_MODEL_CLASS_NAMES: dict[int, str] = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}


def validate_model_class_names(names: Mapping[int, str] | Sequence[str]) -> None:
    """Reject models whose class IDs do not match the approved v1 policy."""

    if isinstance(names, Mapping):
        try:
            actual = {int(index): str(name) for index, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("model class mapping is invalid") from exc
    elif not isinstance(names, (str, bytes)):
        actual = {index: str(name) for index, name in enumerate(names)}
    else:
        raise TypeError("model class mapping is invalid")
    if actual != APPROVED_MODEL_CLASS_NAMES:
        raise ValueError(
            "model class IDs/names do not match the approved Signatus v1 class policy"
        )


class FramePublisher(Protocol):
    def publish(
        self,
        frame: object,
        *,
        timestamp_ns: int | None = None,
        detections: tuple[FrameDetection, ...] = (),
    ) -> bool: ...


class UltralyticsOpenVINOTracker:
    def __init__(
        self,
        model_path: str,
        camera_source: int | str,
        person_class: str,
        lost_timeout_seconds: float,
        cache: DetectionCache,
        publish: Callable[[AIEvent], Awaitable[None]],
        frame_publisher: FramePublisher | None = None,
    ):
        self._model_path = model_path
        self._camera_source = camera_source
        self._person_class = person_class.casefold()
        self._lost_timeout_seconds = lost_timeout_seconds
        self._cache = cache
        self._publish = publish
        self._frame_publisher = frame_publisher
        self._capture = None
        self._model = None
        self._first_frame = True
        self._capture_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="signatus-camera",
        )

    @property
    def model_initialized(self) -> bool:
        return self._model is not None

    @property
    def camera_open(self) -> bool:
        capture = self._capture
        if capture is None:
            return False
        try:
            return bool(capture.isOpened())
        except AttributeError:
            return True

    def initialize_model(self) -> None:
        """Load and validate YOLO exactly once for this AI service process."""

        if self._model is not None:
            return
        from ultralytics import YOLO

        model = YOLO(self._model_path)
        validate_model_class_names(model.names)
        self._model = model

    def open_camera(self) -> None:
        """Open only the configured camera; model initialization is separate."""

        if self._model is None:
            raise RuntimeError("YOLO model is not initialized")
        if self.camera_open:
            return

        import cv2

        capture = cv2.VideoCapture(self._camera_source)
        if not capture.isOpened():
            capture.release()
            raise CameraOperationError(
                f"Unable to open camera source {self._camera_source!r}"
            )
        self._capture = capture
        self._reset_tracking_state()

    async def open_camera_async(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._capture_executor, self.open_camera)

    def _open(self) -> None:
        """Compatibility helper used by focused tracker tests."""

        self.initialize_model()
        self.open_camera()

    def release_camera(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        self._reset_tracking_state()

    def close(self) -> None:
        """Release capture and join the dedicated camera worker."""

        self.release_camera()
        self._capture_executor.shutdown(wait=True, cancel_futures=True)

    def _reset_tracking_state(self) -> None:
        """Discard ByteTrack state without reloading the YOLO model."""

        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()) or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        self._first_frame = True

    def _read_snapshot(self) -> FrameSnapshot:
        if self._capture is None or self._model is None:
            raise RuntimeError("Camera and YOLO model must be ready before capture")

        ok, frame = self._capture.read()
        if not ok:
            raise CameraOperationError("Camera frame read failed")

        captured_ns = time.time_ns()
        result = self._model.track(
            source=frame,
            persist=not self._first_frame,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        self._first_frame = False
        captured_at = time.time()
        people: list[TrackedPerson] = []
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self._publish_preview(frame, captured_ns, ())
            return FrameSnapshot(captured_at, (), (), frame)

        class_ids = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

        for class_id, confidence, coords, track_id in zip(
            class_ids, confidences, coordinates, track_ids, strict=True
        ):
            class_name = str(result.names[int(class_id)])
            box = BoundingBox(*(float(value) for value in coords))
            detections.append(Detection(class_name, float(confidence), box))
            if class_name.casefold() == self._person_class and track_id is not None:
                people.append(TrackedPerson(int(track_id), float(confidence), box))

        self._publish_preview(frame, captured_ns, tuple(detections))
        return FrameSnapshot(captured_at, tuple(people), tuple(detections), frame)

    def _publish_preview(
        self,
        frame: object,
        captured_ns: int,
        detections: tuple[Detection, ...],
    ) -> None:
        if self._frame_publisher is None:
            return
        overlays = tuple(
            FrameDetection(
                detection.class_name,
                detection.confidence,
                detection.box.x1,
                detection.box.y1,
                detection.box.x2,
                detection.box.y2,
            )
            for detection in detections
        )
        try:
            self._frame_publisher.publish(
                frame,
                timestamp_ns=captured_ns,
                detections=overlays,
            )
        except Exception:
            # Live preview is observational. It must never interrupt model
            # inference, event delivery, or the cached decision inputs.
            logger.exception("Camera preview publisher failed; tracking will continue")

    async def run(self, stop: asyncio.Event) -> None:
        last_seen: dict[int, float] = {}
        try:
            while not stop.is_set():
                loop = asyncio.get_running_loop()
                snapshot = await loop.run_in_executor(
                    self._capture_executor,
                    self._read_snapshot,
                )
                if stop.is_set():
                    break
                await self._cache.store(snapshot)
                now = time.monotonic()
                for person in snapshot.people:
                    last_seen[person.track_id] = now
                    await self._publish(
                        AIEvent(
                            type=AIEventType.PERSON_SEEN,
                            track_id=person.track_id,
                            captured_at=snapshot.captured_at,
                        )
                    )

                expired = [
                    track_id
                    for track_id, seen_at in last_seen.items()
                    if now - seen_at >= self._lost_timeout_seconds
                ]
                for track_id in expired:
                    del last_seen[track_id]
                    await self._publish(
                        AIEvent(
                            type=AIEventType.TRACK_LOST,
                            track_id=track_id,
                            captured_at=snapshot.captured_at,
                        )
                    )
        finally:
            captured_at = time.time()
            for track_id in tuple(last_seen):
                await self._publish(
                    AIEvent(
                        type=AIEventType.TRACK_LOST,
                        track_id=track_id,
                        captured_at=captured_at,
                    )
                )
            self.release_camera()
