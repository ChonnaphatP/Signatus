from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, Field

SFACE_DESCRIPTOR_DIMENSIONS: Final = 128
MAX_ENROLLMENT_IMAGE_BYTES: Final = 10 * 1024 * 1024


class CoreState(StrEnum):
    STANDBY = "STANDBY"
    AUTHORIZATION = "AUTHORIZATION"


class AIServiceState(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class CameraState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class ValidationSeverity(StrEnum):
    FATAL = "FATAL"
    DATA_ERROR = "DATA_ERROR"
    WARNING = "WARNING"


class AIEventType(StrEnum):
    PERSON_SEEN = "PERSON_SEEN"
    TRACK_LOST = "TRACK_LOST"


class OutcomeStatus(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    PPE_VIOLATION = "PPE_VIOLATION"
    UNAUTHORIZED = "UNAUTHORIZED"
    FACE_CAPTURE_FAILED = "FACE_CAPTURE_FAILED"


class FaceEmbeddingStatus(StrEnum):
    OK = "OK"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_QUALITY = "LOW_QUALITY"
    ERROR = "ERROR"


class PPEResultStatus(StrEnum):
    OK = "OK"
    NO_CACHED_FRAME = "NO_CACHED_FRAME"
    TRACK_NOT_FOUND = "TRACK_NOT_FOUND"
    ASSOCIATION_UNRESOLVED = "ASSOCIATION_UNRESOLVED"
    ERROR = "ERROR"


class AIEvent(BaseModel):
    type: AIEventType
    track_id: int = Field(ge=0)
    captured_at: float


class EmbeddingResult(BaseModel):
    track_id: int = Field(ge=0)
    status: FaceEmbeddingStatus
    embedding: list[float] | None = None


class PPEResult(BaseModel):
    track_id: int = Field(ge=0)
    status: PPEResultStatus = PPEResultStatus.OK
    detected_classes: list[str] = Field(default_factory=list)
    captured_at: float | None = None


class CameraStatus(BaseModel):
    state: CameraState
    error: str | None = None


class ValidationIssue(BaseModel):
    severity: ValidationSeverity
    code: str
    message: str
    worksite_id: str | None = None
    worker_id: str | None = None
    source: str | None = None


class ValidationReport(BaseModel):
    issues: list[ValidationIssue] = Field(default_factory=list)


class GUIStatusSignal(BaseModel):
    status: OutcomeStatus
    worker_id: str | None = None
    missing_ppe: list[str] | None = None
    face_failure_reason: FaceEmbeddingStatus | None = None
    attempt: int | None = None
    retry_allowed: bool | None = None
