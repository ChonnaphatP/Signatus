from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _parse_camera_source(value: str) -> int | str:
    stripped = value.strip()
    return int(stripped) if stripped.isdecimal() else stripped


@dataclass(frozen=True, slots=True)
class AISettings:
    model_path: Path
    camera_source: int | str
    person_class: str
    track_lost_timeout_seconds: float
    tracking_enabled: bool
    ppe_association: str
    face_detector_model_path: Path
    face_recognizer_model_path: Path
    face_detector_score_threshold: float
    face_detector_nms_threshold: float
    face_detector_top_k: int
    frame_shm_enabled: bool
    frame_shm_name: str
    preview_max_frame_bytes: int

    @classmethod
    def from_environment(cls) -> AISettings:
        return cls(
            model_path=Path(
                os.getenv(
                    "SIGNATUS_MODEL_PATH",
                    "./models/yolo26s100e18b_int8_openvino_model",
                )
            ),
            camera_source=_parse_camera_source(os.getenv("SIGNATUS_CAMERA_SOURCE", "0")),
            person_class=os.getenv("SIGNATUS_PERSON_CLASS", "Person"),
            track_lost_timeout_seconds=float(
                os.getenv("SIGNATUS_TRACK_LOST_TIMEOUT_SECONDS", "1.5")
            ),
            tracking_enabled=_parse_bool(os.getenv("SIGNATUS_AI_TRACKING_ENABLED", "false")),
            ppe_association=os.getenv("SIGNATUS_PPE_ASSOCIATION", "single_person_frame"),
            face_detector_model_path=Path(
                os.getenv(
                    "SIGNATUS_FACE_DETECTOR_MODEL_PATH",
                    "./models/face_detection_yunet_2023mar.onnx",
                )
            ),
            face_recognizer_model_path=Path(
                os.getenv(
                    "SIGNATUS_FACE_RECOGNIZER_MODEL_PATH",
                    "./models/face_recognition_sface_2021dec.onnx",
                )
            ),
            face_detector_score_threshold=float(
                os.getenv("SIGNATUS_FACE_DETECTOR_SCORE_THRESHOLD", "0.9")
            ),
            face_detector_nms_threshold=float(
                os.getenv("SIGNATUS_FACE_DETECTOR_NMS_THRESHOLD", "0.3")
            ),
            face_detector_top_k=int(os.getenv("SIGNATUS_FACE_DETECTOR_TOP_K", "5000")),
            frame_shm_enabled=_parse_bool(os.getenv("SIGNATUS_FRAME_SHM_ENABLED", "true")),
            frame_shm_name=os.getenv("SIGNATUS_FRAME_SHM_NAME", "signatus_camera_v1").strip(),
            preview_max_frame_bytes=int(os.getenv("SIGNATUS_PREVIEW_MAX_FRAME_BYTES", "6220800")),
        )
