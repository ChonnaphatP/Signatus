from __future__ import annotations

import base64
import binascii
import logging
import math
import re
import threading
from pathlib import Path
from typing import Final, Protocol

import cv2
import numpy as np

from signatus_contracts import (
    MAX_ENROLLMENT_IMAGE_BYTES,
    SFACE_DESCRIPTOR_DIMENSIONS,
    EmbeddingResult,
    FaceEmbeddingStatus,
)

from .cache import BoundingBox, FrameSnapshot

logger = logging.getLogger(__name__)

ENROLLMENT_EMBEDDING_TRACK_ID: Final = 0
MAX_ENROLLMENT_IMAGE_PIXELS: Final = 25_000_000
_MAX_ENCODED_IMAGE_CHARACTERS: Final = ((MAX_ENROLLMENT_IMAGE_BYTES + 2) // 3) * 4
_FACE_IMAGE_DATA_URI = re.compile(
    r"\Adata:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]*={0,2})\Z"
)


class FaceEmbedder(Protocol):
    async def generate(self, track_id: int, snapshot: FrameSnapshot | None) -> EmbeddingResult: ...

    async def generate_face_image(self, face_image: str) -> EmbeddingResult: ...


class OpenCVYuNetSFaceEmbedder:
    """Extract SFace FP32 embeddings from YuNet-aligned tracked-person faces."""

    def __init__(
        self,
        detector_model_path: Path,
        recognizer_model_path: Path,
        *,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self._detector_model_path = detector_model_path
        self._recognizer_model_path = recognizer_model_path
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._detector: object | None = None
        self._recognizer: object | None = None
        self._lock = threading.Lock()

    @property
    def model_files_available(self) -> bool:
        return self._detector_model_path.is_file() and self._recognizer_model_path.is_file()

    @property
    def models_initialized(self) -> bool:
        return self._detector is not None and self._recognizer is not None

    def initialize(self) -> None:
        """Load both approved face models or raise before service readiness."""

        with self._lock:
            self._ensure_models(cv2)

    async def generate(self, track_id: int, snapshot: FrameSnapshot | None) -> EmbeddingResult:
        if snapshot is None:
            return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.NO_FACE)
        person = next((item for item in snapshot.people if item.track_id == track_id), None)
        if person is None:
            return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.NO_FACE)
        if snapshot.frame is None:
            return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)
        try:
            with self._lock:
                self._ensure_models(cv2)
        except (OSError, ValueError, cv2.error):
            logger.exception("YuNet/SFace model initialization failed")
            return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)
        # These OpenCV DNN objects are synchronous and guarded as a single-model
        # resource. Keeping inference on the service thread avoids platform-specific
        # executor hangs observed during model-backed startup validation.
        return self._generate_sync(track_id, snapshot.frame, person.box)

    async def generate_face_image(self, face_image: str) -> EmbeddingResult:
        """Embed one enrollment image without depending on camera or track state."""

        try:
            image = _decode_face_image_data_uri(face_image)
        except (TypeError, ValueError) as error:
            logger.warning("Worker Profile face image was rejected: %s", error)
            return EmbeddingResult(
                track_id=ENROLLMENT_EMBEDDING_TRACK_ID,
                status=FaceEmbeddingStatus.ERROR,
            )
        return self._generate_image_sync(ENROLLMENT_EMBEDDING_TRACK_ID, image)

    def close(self) -> None:
        """Release hook retained for the service lifecycle."""

    def _generate_sync(
        self,
        track_id: int,
        frame: object,
        person_box: BoundingBox,
    ) -> EmbeddingResult:
        try:
            image = np.asarray(frame)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)
            crop = _crop_person(image, person_box)
            if crop is None:
                return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.NO_FACE)
        except (OSError, TypeError, ValueError, cv2.error):
            logger.exception("YuNet/SFace embedding failed for track %d", track_id)
            return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)
        return self._generate_image_sync(track_id, crop)

    def _generate_image_sync(self, track_id: int, image: np.ndarray) -> EmbeddingResult:
        try:
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)
            if image.shape[0] <= 0 or image.shape[1] <= 0:
                return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)

            contiguous_image = np.ascontiguousarray(image)
            with self._lock:
                self._ensure_models(cv2)
                detector = self._detector
                recognizer = self._recognizer
                if detector is None or recognizer is None:
                    return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)

                detector.setInputSize((contiguous_image.shape[1], contiguous_image.shape[0]))
                _result, faces = detector.detect(contiguous_image)
                if faces is None or len(faces) == 0:
                    return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.NO_FACE)
                if len(faces) != 1:
                    return EmbeddingResult(
                        track_id=track_id,
                        status=FaceEmbeddingStatus.MULTIPLE_FACES,
                    )

                aligned = recognizer.alignCrop(contiguous_image, faces[0])
                feature = np.asarray(recognizer.feature(aligned), dtype=np.float32).reshape(-1)

            values = tuple(float(value) for value in feature)
            norm = math.sqrt(sum(value * value for value in values))
            if len(values) != SFACE_DESCRIPTOR_DIMENSIONS:
                return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.LOW_QUALITY)
            if not math.isfinite(norm) or norm == 0.0:
                return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.LOW_QUALITY)
            if any(not math.isfinite(value) for value in values):
                return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.LOW_QUALITY)
            return EmbeddingResult(
                track_id=track_id,
                status=FaceEmbeddingStatus.OK,
                embedding=list(values),
            )
        except (OSError, TypeError, ValueError, cv2.error):
            logger.exception("YuNet/SFace embedding failed")
            return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)

    def _ensure_models(self, cv2: object) -> None:
        if self._detector is not None and self._recognizer is not None:
            return
        if not self.model_files_available:
            raise OSError("YuNet or SFace model file is unavailable")
        self._detector = cv2.FaceDetectorYN.create(
            str(self._detector_model_path),
            "",
            (320, 320),
            self._score_threshold,
            self._nms_threshold,
            self._top_k,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            str(self._recognizer_model_path),
            "",
        )


def _crop_person(image: np.ndarray, box: BoundingBox) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1 = max(0, min(width, math.floor(box.x1)))
    y1 = max(0, min(height, math.floor(box.y1)))
    x2 = max(0, min(width, math.ceil(box.x2)))
    y2 = max(0, min(height, math.ceil(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def _decode_face_image_data_uri(data_uri: str) -> np.ndarray:
    if not isinstance(data_uri, str):
        raise TypeError("face_image must be a string data URI")
    if len(data_uri) > _MAX_ENCODED_IMAGE_CHARACTERS + 64:
        raise ValueError(f"decoded face_image must not exceed {MAX_ENROLLMENT_IMAGE_BYTES} bytes")
    match = _FACE_IMAGE_DATA_URI.fullmatch(data_uri)
    if match is None:
        raise ValueError("face_image must be a complete JPEG, PNG, or WebP base64 data URI")
    declared_mime_type, encoded = match.groups()
    if not encoded:
        raise ValueError("face_image contains no image data")
    if len(encoded) > _MAX_ENCODED_IMAGE_CHARACTERS:
        raise ValueError(f"decoded face_image must not exceed {MAX_ENROLLMENT_IMAGE_BYTES} bytes")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("face_image contains invalid base64 data") from error
    if not image_bytes:
        raise ValueError("face_image contains no image data")
    if len(image_bytes) > MAX_ENROLLMENT_IMAGE_BYTES:
        raise ValueError(f"decoded face_image must not exceed {MAX_ENROLLMENT_IMAGE_BYTES} bytes")
    detected_mime_type = _detect_image_mime_type(image_bytes)
    if detected_mime_type is None:
        raise ValueError("face_image data is not a supported JPEG, PNG, or WebP image")
    if detected_mime_type != declared_mime_type:
        raise ValueError("face_image MIME type does not match its image content")
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError("face_image could not be decoded") from error
    if image is None:
        raise ValueError("face_image could not be decoded")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("face_image did not decode to an 8-bit three-channel image")
    if image.shape[0] * image.shape[1] > MAX_ENROLLMENT_IMAGE_PIXELS:
        raise ValueError("face_image dimensions exceed the enrollment image limit")
    return np.ascontiguousarray(image)


def _detect_image_mime_type(image_bytes: bytes) -> str | None:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None
