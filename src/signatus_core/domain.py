from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CoreState(StrEnum):
    STANDBY = "STANDBY"
    AUTHORIZATION = "AUTHORIZATION"


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


@dataclass(frozen=True, slots=True)
class AIEvent:
    type: AIEventType
    track_id: int
    captured_at: float


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    track_id: int
    status: FaceEmbeddingStatus
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class PPEResult:
    track_id: int
    status: PPEResultStatus
    detected_classes: tuple[str, ...] = ()
    captured_at: float | None = None


@dataclass(frozen=True, slots=True)
class GUIStatusSignal:
    status: OutcomeStatus
    worker_id: str | None = None
    missing_ppe: tuple[str, ...] | None = None
    face_failure_reason: FaceEmbeddingStatus | None = None
    attempt: int | None = None
    retry_allowed: bool | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"status": self.status.value}
        if self.worker_id is not None:
            payload["worker_id"] = self.worker_id
        if self.missing_ppe is not None:
            payload["missing_ppe"] = list(self.missing_ppe)
        if self.face_failure_reason is not None:
            payload["face_failure_reason"] = self.face_failure_reason.value
        if self.attempt is not None:
            payload["attempt"] = self.attempt
        if self.retry_allowed is not None:
            payload["retry_allowed"] = self.retry_allowed
        return payload


@dataclass(frozen=True, slots=True)
class AuthorizedWorker:
    worker_id: str
    embedding: tuple[float, ...]
    name: str = ""


@dataclass(frozen=True, slots=True)
class Worksite:
    worksite_id: str
    name: str
    authorized_workers: tuple[AuthorizedWorker, ...]
    required_ppe: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PPEClassRule:
    positive_classes: frozenset[str]
    negative_classes: frozenset[str]


@dataclass(frozen=True, slots=True)
class PPEEvaluation:
    compliant: bool
    missing_ppe: tuple[str, ...]
