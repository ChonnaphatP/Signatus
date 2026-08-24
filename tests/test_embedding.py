from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from signatus_ai.cache import BoundingBox, FrameSnapshot, TrackedPerson
from signatus_ai.embedding import (
    ENROLLMENT_EMBEDDING_TRACK_ID,
    OpenCVYuNetSFaceEmbedder,
)
from signatus_contracts import FaceEmbeddingStatus

BOX = BoundingBox(2.0, 1.0, 8.0, 7.0)


class FakeDetector:
    def __init__(self, faces: np.ndarray | None) -> None:
        self.faces = faces
        self.input_size: tuple[int, int] | None = None
        self.detected_shape: tuple[int, ...] | None = None

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_size = size

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        self.detected_shape = image.shape
        return 1, self.faces


class FakeRecognizer:
    def __init__(self, feature: np.ndarray | None = None) -> None:
        default = np.zeros((1, 128), np.float32)
        default[0, :2] = (3.0, 4.0)
        self.output = feature if feature is not None else default
        self.aligned_shape: tuple[int, ...] | None = None

    def alignCrop(self, image: np.ndarray, _face: np.ndarray) -> np.ndarray:
        self.aligned_shape = image.shape
        return image

    def feature(self, _aligned: np.ndarray) -> np.ndarray:
        return self.output


def snapshot(*, track_id: int = 7, frame: object | None = None) -> FrameSnapshot:
    return FrameSnapshot(
        captured_at=1000.0,
        people=(TrackedPerson(track_id, 0.9, BOX),),
        detections=(),
        frame=np.zeros((10, 12, 3), dtype=np.uint8) if frame is None else frame,
    )


def embedder_with(
    faces: np.ndarray | None,
    *,
    feature: np.ndarray | None = None,
) -> tuple[OpenCVYuNetSFaceEmbedder, FakeDetector, FakeRecognizer]:
    embedder = OpenCVYuNetSFaceEmbedder(Path("unused-yunet"), Path("unused-sface"))
    detector = FakeDetector(faces)
    recognizer = FakeRecognizer(feature)
    embedder._detector = detector
    embedder._recognizer = recognizer
    return embedder, detector, recognizer


def face_image_data_uri(
    *,
    height: int = 14,
    width: int = 20,
    extension: str = ".png",
    mime_type: str = "image/png",
) -> str:
    success, encoded = cv2.imencode(
        extension,
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    if not success:
        raise AssertionError("OpenCV test image encoding failed")
    return f"data:{mime_type};base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


class OpenCVYuNetSFaceEmbedderTests(unittest.IsolatedAsyncioTestCase):
    def test_initialize_loads_both_face_models_for_readiness(self) -> None:
        detector = object()
        recognizer = object()
        fake_cv2 = SimpleNamespace(
            FaceDetectorYN=SimpleNamespace(create=lambda *_args: detector),
            FaceRecognizerSF=SimpleNamespace(create=lambda *_args: recognizer),
        )
        with tempfile.TemporaryDirectory() as directory:
            detector_path = Path(directory) / "yunet.onnx"
            recognizer_path = Path(directory) / "sface.onnx"
            detector_path.write_bytes(b"model")
            recognizer_path.write_bytes(b"model")
            embedder = OpenCVYuNetSFaceEmbedder(detector_path, recognizer_path)
            with patch("signatus_ai.embedding.cv2", fake_cv2):
                embedder.initialize()

        self.assertTrue(embedder.models_initialized)

    async def test_missing_snapshot_or_track_returns_no_face(self) -> None:
        embedder, _detector, _recognizer = embedder_with(np.zeros((1, 15), np.float32))
        self.addCleanup(embedder.close)

        no_snapshot = await embedder.generate(7, None)
        wrong_track = await embedder.generate(8, snapshot())

        self.assertEqual(no_snapshot.status, FaceEmbeddingStatus.NO_FACE)
        self.assertEqual(wrong_track.status, FaceEmbeddingStatus.NO_FACE)

    async def test_missing_cached_pixels_returns_error(self) -> None:
        embedder, _detector, _recognizer = embedder_with(np.zeros((1, 15), np.float32))
        self.addCleanup(embedder.close)
        cached = FrameSnapshot(1000.0, (TrackedPerson(7, 0.9, BOX),), (), None)

        result = await embedder.generate(7, cached)

        self.assertEqual(result.status, FaceEmbeddingStatus.ERROR)

    async def test_no_or_multiple_faces_return_distinct_failures(self) -> None:
        no_face, _detector, _recognizer = embedder_with(None)
        multiple, _detector2, _recognizer2 = embedder_with(np.zeros((2, 15), np.float32))
        self.addCleanup(no_face.close)
        self.addCleanup(multiple.close)

        no_face_result = await no_face.generate(7, snapshot())
        multiple_result = await multiple.generate(7, snapshot())

        self.assertEqual(no_face_result.status, FaceEmbeddingStatus.NO_FACE)
        self.assertEqual(multiple_result.status, FaceEmbeddingStatus.MULTIPLE_FACES)

    async def test_person_crop_is_aligned_and_embedded(self) -> None:
        embedder, detector, recognizer = embedder_with(np.zeros((1, 15), np.float32))
        self.addCleanup(embedder.close)

        result = await embedder.generate(7, snapshot())

        self.assertEqual(result.status, FaceEmbeddingStatus.OK)
        self.assertEqual(result.embedding[:2], [3.0, 4.0])
        self.assertEqual(len(result.embedding), 128)
        self.assertEqual(detector.input_size, (6, 6))
        self.assertEqual(detector.detected_shape, (6, 6, 3))
        self.assertEqual(recognizer.aligned_shape, (6, 6, 3))

    async def test_non_finite_or_zero_feature_fails_quality(self) -> None:
        zero, _detector, _recognizer = embedder_with(
            np.zeros((1, 15), np.float32),
            feature=np.zeros((1, 128), np.float32),
        )
        non_finite, _detector2, _recognizer2 = embedder_with(
            np.zeros((1, 15), np.float32),
            feature=np.array([[np.nan] + [1.0] * 127], np.float32),
        )
        self.addCleanup(zero.close)
        self.addCleanup(non_finite.close)

        zero_result = await zero.generate(7, snapshot())
        non_finite_result = await non_finite.generate(7, snapshot())

        self.assertEqual(zero_result.status, FaceEmbeddingStatus.LOW_QUALITY)
        self.assertEqual(non_finite_result.status, FaceEmbeddingStatus.LOW_QUALITY)

    async def test_wrong_sface_descriptor_size_fails_quality(self) -> None:
        embedder, _detector, _recognizer = embedder_with(
            np.zeros((1, 15), np.float32),
            feature=np.array([[3.0, 4.0]], np.float32),
        )
        self.addCleanup(embedder.close)

        result = await embedder.generate(7, snapshot())

        self.assertEqual(result.status, FaceEmbeddingStatus.LOW_QUALITY)

    async def test_enrollment_data_uri_embeds_the_complete_image(self) -> None:
        embedder, detector, recognizer = embedder_with(np.zeros((1, 15), np.float32))
        self.addCleanup(embedder.close)

        result = await embedder.generate_face_image(face_image_data_uri())

        self.assertEqual(result.track_id, ENROLLMENT_EMBEDDING_TRACK_ID)
        self.assertEqual(result.status, FaceEmbeddingStatus.OK)
        self.assertEqual(len(result.embedding or ()), 128)
        self.assertEqual(detector.input_size, (20, 14))
        self.assertEqual(detector.detected_shape, (14, 20, 3))
        self.assertEqual(recognizer.aligned_shape, (14, 20, 3))

    async def test_enrollment_accepts_jpeg_png_and_webp_data_uris(self) -> None:
        embedder, _detector, _recognizer = embedder_with(np.zeros((1, 15), np.float32))
        self.addCleanup(embedder.close)

        for extension, mime_type in (
            (".jpg", "image/jpeg"),
            (".png", "image/png"),
            (".webp", "image/webp"),
        ):
            with self.subTest(mime_type=mime_type):
                result = await embedder.generate_face_image(
                    face_image_data_uri(extension=extension, mime_type=mime_type)
                )
                self.assertEqual(result.status, FaceEmbeddingStatus.OK)

    async def test_enrollment_preserves_face_and_descriptor_failure_statuses(self) -> None:
        no_face, _detector, _recognizer = embedder_with(None)
        multiple, _detector2, _recognizer2 = embedder_with(np.zeros((2, 15), np.float32))
        low_quality, _detector3, _recognizer3 = embedder_with(
            np.zeros((1, 15), np.float32),
            feature=np.zeros((1, 128), np.float32),
        )
        self.addCleanup(no_face.close)
        self.addCleanup(multiple.close)
        self.addCleanup(low_quality.close)
        image = face_image_data_uri()

        no_face_result = await no_face.generate_face_image(image)
        multiple_result = await multiple.generate_face_image(image)
        low_quality_result = await low_quality.generate_face_image(image)

        self.assertEqual(no_face_result.status, FaceEmbeddingStatus.NO_FACE)
        self.assertEqual(multiple_result.status, FaceEmbeddingStatus.MULTIPLE_FACES)
        self.assertEqual(low_quality_result.status, FaceEmbeddingStatus.LOW_QUALITY)

    async def test_enrollment_rejects_invalid_mismatched_and_oversize_data_uris(self) -> None:
        embedder, detector, _recognizer = embedder_with(np.zeros((1, 15), np.float32))
        self.addCleanup(embedder.close)
        png_declared_as_jpeg = face_image_data_uri(mime_type="image/jpeg")

        invalid = await embedder.generate_face_image("not-a-data-uri")
        mismatch = await embedder.generate_face_image(png_declared_as_jpeg)
        with patch("signatus_ai.embedding._MAX_ENCODED_IMAGE_CHARACTERS", 8):
            oversize = await embedder.generate_face_image(
                "data:image/png;base64," + "A" * 100
            )

        self.assertEqual(invalid.status, FaceEmbeddingStatus.ERROR)
        self.assertEqual(mismatch.status, FaceEmbeddingStatus.ERROR)
        self.assertEqual(oversize.status, FaceEmbeddingStatus.ERROR)
        self.assertIsNone(detector.detected_shape)

    async def test_enrollment_reuses_initialized_models_across_requests(self) -> None:
        embedder, detector, recognizer = embedder_with(np.zeros((1, 15), np.float32))
        self.addCleanup(embedder.close)
        original_detector = embedder._detector
        original_recognizer = embedder._recognizer
        image = face_image_data_uri()

        first = await embedder.generate_face_image(image)
        second = await embedder.generate_face_image(image)

        self.assertEqual(first.status, FaceEmbeddingStatus.OK)
        self.assertEqual(second.status, FaceEmbeddingStatus.OK)
        self.assertIs(embedder._detector, original_detector)
        self.assertIs(embedder._recognizer, original_recognizer)
        self.assertIs(embedder._detector, detector)
        self.assertIs(embedder._recognizer, recognizer)


if __name__ == "__main__":
    unittest.main()
