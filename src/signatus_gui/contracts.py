from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class OutcomeStatus(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    PPE_VIOLATION = "PPE_VIOLATION"
    UNAUTHORIZED = "UNAUTHORIZED"
    FACE_CAPTURE_FAILED = "FACE_CAPTURE_FAILED"


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


class GUIProtocolError(ValueError):
    """Raised when Core sends a payload outside the established GUI contract."""


@dataclass(frozen=True, slots=True)
class Worksite:
    worksite_id: str
    name: str
    required_ppe: tuple[str, ...] = ()
    available: bool = True
    unavailable_reason: str | None = None
    valid_worker_count: int = 0
    invalid_worker_count: int = 0


@dataclass(frozen=True, slots=True)
class ManagedWorksite:
    source: str
    worksite_id: str | None
    name: str | None
    required_ppe: tuple[str, ...]
    available: bool
    unavailable_reason: str | None
    valid_worker_count: int
    invalid_worker_count: int
    issues: tuple[ValidationIssue, ...] = ()
    active: bool = False


@dataclass(frozen=True, slots=True)
class WorksiteWorker:
    worker_id: str
    name: str
    embedding: tuple[float, ...] | None = None
    face_image: str | None = None

    def __post_init__(self) -> None:
        if (self.embedding is None) == (self.face_image is None):
            raise ValueError(
                "A Wo.No. worker must contain exactly one of embedding or face_image"
            )


@dataclass(frozen=True, slots=True)
class WorksiteDraft:
    worksite_id: str
    name: str
    authorized_workers: tuple[WorksiteWorker, ...]
    required_ppe: tuple[str, ...]
    source: str | None = None
    invalid_worker_messages: tuple[str, ...] = ()
    active: bool = False


@dataclass(frozen=True, slots=True)
class RawJsonDocument:
    source: str
    raw: str
    formatted: str
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class ImportSummary:
    imported: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    worker_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkerProfile:
    worker_id: str
    name: str
    face_image: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class CameraStatus:
    state: CameraState
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    worksite_id: str | None = None
    worker_id: str | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def data_errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is ValidationSeverity.DATA_ERROR
        )


@dataclass(frozen=True, slots=True)
class Outcome:
    status: OutcomeStatus
    worker_id: str | None = None
    missing_ppe: tuple[str, ...] = ()
    face_failure_reason: str | None = None
    attempt: int | None = None
    retry_allowed: bool | None = None


def parse_worksites(payload: Any) -> tuple[Worksite, ...]:
    if not isinstance(payload, list):
        raise GUIProtocolError("Core returned an invalid worksite list")

    worksites: list[Worksite] = []
    for item in payload:
        if not isinstance(item, dict):
            raise GUIProtocolError("Core returned an invalid worksite")
        worksite_id = item.get("worksite_id")
        name = item.get("name")
        required_ppe = item.get("required_ppe", [])
        available = item.get("available", True)
        unavailable_reason = item.get("unavailable_reason")
        valid_worker_count = item.get("valid_worker_count", 0)
        invalid_worker_count = item.get("invalid_worker_count", 0)
        if not isinstance(worksite_id, str) or not worksite_id.strip():
            raise GUIProtocolError("Core returned a worksite without a Wo.No.")
        if not isinstance(name, str) or not name.strip():
            raise GUIProtocolError("Core returned a worksite without a name")
        if not isinstance(required_ppe, list) or any(
            not isinstance(value, str) or not value.strip() for value in required_ppe
        ):
            raise GUIProtocolError("Core returned an invalid required-PPE list")
        if not isinstance(available, bool):
            raise GUIProtocolError("Core returned an invalid worksite availability")
        if unavailable_reason is not None and not isinstance(unavailable_reason, str):
            raise GUIProtocolError("Core returned an invalid unavailability reason")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (valid_worker_count, invalid_worker_count)
        ):
            raise GUIProtocolError("Core returned invalid worker validation counts")
        worksites.append(
            Worksite(
                worksite_id=worksite_id,
                name=name,
                required_ppe=tuple(required_ppe),
                available=available,
                unavailable_reason=unavailable_reason,
                valid_worker_count=valid_worker_count,
                invalid_worker_count=invalid_worker_count,
            )
        )
    return tuple(worksites)


def parse_manager_catalog(payload: Any) -> tuple[ManagedWorksite, ...]:
    if not isinstance(payload, list):
        raise GUIProtocolError("Core returned an invalid Manager catalog")
    entries: list[ManagedWorksite] = []
    for item in payload:
        if not isinstance(item, dict):
            raise GUIProtocolError("Core returned an invalid Manager entry")
        source = item.get("source")
        worksite_id = item.get("worksite_id")
        name = item.get("name")
        required_ppe = item.get("required_ppe", [])
        available = item.get("available")
        unavailable_reason = item.get("unavailable_reason")
        valid_count = item.get("valid_worker_count", 0)
        invalid_count = item.get("invalid_worker_count", 0)
        active = item.get("active", False)
        if not isinstance(source, str) or not source:
            raise GUIProtocolError("Core returned a Manager entry without a source")
        if worksite_id is not None and not isinstance(worksite_id, str):
            raise GUIProtocolError("Core returned an invalid Manager Wo.No.")
        if name is not None and not isinstance(name, str):
            raise GUIProtocolError("Core returned an invalid Manager worksite name")
        if not isinstance(required_ppe, list) or any(
            not isinstance(value, str) for value in required_ppe
        ):
            raise GUIProtocolError("Core returned invalid Manager PPE data")
        if not isinstance(available, bool) or not isinstance(active, bool):
            raise GUIProtocolError("Core returned invalid Manager state")
        if unavailable_reason is not None and not isinstance(unavailable_reason, str):
            raise GUIProtocolError("Core returned an invalid Manager reason")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (valid_count, invalid_count)
        ):
            raise GUIProtocolError("Core returned invalid Manager worker counts")
        issue_payload = item.get("issues", [])
        issues = parse_validation_report({"issues": issue_payload}).issues
        entries.append(
            ManagedWorksite(
                source=source,
                worksite_id=worksite_id,
                name=name,
                required_ppe=tuple(required_ppe),
                available=available,
                unavailable_reason=unavailable_reason,
                valid_worker_count=valid_count,
                invalid_worker_count=invalid_count,
                issues=issues,
                active=active,
            )
        )
    return tuple(entries)


def parse_worksite_draft(payload: Any) -> WorksiteDraft:
    if not isinstance(payload, dict):
        raise GUIProtocolError("Core returned invalid Wo.No. editor data")
    worksite_id = payload.get("worksite_id")
    name = payload.get("name")
    workers = payload.get("authorized_workers")
    required_ppe = payload.get("required_ppe")
    source = payload.get("source")
    invalid_messages = payload.get("invalid_worker_messages", [])
    active = payload.get("active", False)
    if not isinstance(worksite_id, str) or not worksite_id.strip():
        raise GUIProtocolError("Core returned editor data without a Wo.No.")
    if not isinstance(name, str) or not name.strip():
        raise GUIProtocolError("Core returned editor data without a name")
    if source is not None and not isinstance(source, str):
        raise GUIProtocolError("Core returned an invalid editor source")
    if not isinstance(required_ppe, list) or any(
        not isinstance(item, str) for item in required_ppe
    ):
        raise GUIProtocolError("Core returned invalid editor PPE data")
    if not isinstance(invalid_messages, list) or any(
        not isinstance(item, str) for item in invalid_messages
    ):
        raise GUIProtocolError("Core returned invalid skipped-worker details")
    if not isinstance(active, bool):
        raise GUIProtocolError("Core returned invalid active-policy state")
    if not isinstance(workers, list):
        raise GUIProtocolError("Core returned invalid editor worker data")
    parsed_workers = tuple(parse_worksite_worker(item) for item in workers)
    return WorksiteDraft(
        worksite_id=worksite_id,
        name=name,
        authorized_workers=parsed_workers,
        required_ppe=tuple(required_ppe),
        source=source,
        invalid_worker_messages=tuple(invalid_messages),
        active=active,
    )


def parse_raw_json_document(payload: Any) -> RawJsonDocument:
    if not isinstance(payload, dict):
        raise GUIProtocolError("Core returned invalid JSON-viewer data")
    source = payload.get("source")
    raw = payload.get("raw")
    formatted = payload.get("formatted")
    parse_error = payload.get("parse_error")
    if not isinstance(source, str) or not isinstance(raw, str):
        raise GUIProtocolError("Core returned incomplete JSON-viewer data")
    if formatted is not None and not isinstance(formatted, str):
        raise GUIProtocolError("Core returned invalid formatted JSON-viewer data")
    if parse_error is not None and not isinstance(parse_error, str):
        raise GUIProtocolError("Core returned an invalid JSON parsing error")
    return RawJsonDocument(source, raw, raw if formatted is None else formatted, parse_error)


def parse_import_summary(payload: Any) -> ImportSummary:
    if not isinstance(payload, dict):
        raise GUIProtocolError("Core returned an invalid import result")

    def descriptions(field: str) -> tuple[str, ...]:
        value = payload.get(field, [])
        if not isinstance(value, list):
            raise GUIProtocolError(f"Core returned invalid import {field}")
        descriptions: list[str] = []
        for item in value:
            if isinstance(item, str):
                descriptions.append(item)
                continue
            if not isinstance(item, dict):
                raise GUIProtocolError(f"Core returned invalid import {field}")
            identity = item.get("worksite_id") or item.get("source") or "Wo.No."
            message = item.get("message", "")
            if not isinstance(identity, str) or not isinstance(message, str):
                raise GUIProtocolError(f"Core returned invalid import {field}")
            descriptions.append(f"{identity}: {message}" if message else identity)
        return tuple(descriptions)

    warnings: list[str] = []
    for item in payload.get("imported", []):
        if not isinstance(item, dict):
            continue
        identity = item.get("worksite_id") or item.get("source") or "Wo.No."
        for issue in item.get("skipped_workers", []):
            if isinstance(issue, dict) and isinstance(issue.get("message"), str):
                worker = issue.get("worker_id")
                prefix = f"{identity} / {worker}" if isinstance(worker, str) else str(identity)
                warnings.append(f"{prefix}: {issue['message']}")

    return ImportSummary(
        imported=descriptions("imported"),
        skipped=descriptions("skipped"),
        failed=descriptions("failed"),
        worker_warnings=tuple(warnings),
    )


def parse_worker_profile(payload: Any) -> WorkerProfile:
    if not isinstance(payload, dict):
        raise GUIProtocolError("Invalid Worker Profile JSON")
    worker_id = payload.get("worker_id")
    name = payload.get("name")
    face_image = payload.get("face_image")
    source = payload.get("source")
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise GUIProtocolError("Worker Profile is missing worker_id")
    if not isinstance(name, str) or not name.strip():
        raise GUIProtocolError("Worker Profile is missing name")
    if not isinstance(face_image, str) or not face_image.startswith("data:image/"):
        raise GUIProtocolError("Worker Profile is missing a complete face-image data URI")
    if source is not None and not isinstance(source, str):
        raise GUIProtocolError("Worker Profile has an invalid source")
    return WorkerProfile(worker_id, name, face_image, source)


def parse_worksite_worker(payload: Any) -> WorksiteWorker:
    if not isinstance(payload, dict):
        raise GUIProtocolError("Core returned an invalid worker")
    has_embedding = payload.get("embedding") is not None
    has_face_image = payload.get("face_image") is not None
    if has_embedding == has_face_image:
        raise GUIProtocolError(
            "Wo.No. worker must contain exactly one of embedding or face_image"
        )
    if has_face_image:
        profile = parse_worker_profile(payload)
        return WorksiteWorker(
            profile.worker_id,
            profile.name,
            face_image=profile.face_image,
        )

    worker_id = payload.get("worker_id")
    name = payload.get("name")
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise GUIProtocolError("Wo.No. worker is missing worker_id")
    if not isinstance(name, str) or not name.strip():
        raise GUIProtocolError("Wo.No. worker is missing name")
    embedding = payload.get("embedding")
    if (
        not isinstance(embedding, list)
        or len(embedding) != 128
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in embedding
        )
    ):
        raise GUIProtocolError("Wo.No. worker embedding must contain 128 finite numbers")
    values = tuple(float(value) for value in embedding)
    if math.hypot(*values) == 0.0:
        raise GUIProtocolError("Wo.No. worker embedding must have a nonzero norm")
    return WorksiteWorker(worker_id, name, embedding=values)


def parse_camera_status(payload: Any) -> CameraStatus:
    if not isinstance(payload, dict):
        raise GUIProtocolError("Core returned an invalid camera status")
    try:
        state = CameraState(payload.get("state"))
    except (TypeError, ValueError) as error:
        raise GUIProtocolError("Core returned an unknown camera state") from error
    camera_error = payload.get("error")
    if camera_error is not None and not isinstance(camera_error, str):
        raise GUIProtocolError("Core returned an invalid camera error")
    return CameraStatus(state=state, error=camera_error)


def parse_validation_report(payload: Any) -> ValidationReport:
    if not isinstance(payload, dict) or not isinstance(payload.get("issues"), list):
        raise GUIProtocolError("Core returned an invalid validation report")
    issues: list[ValidationIssue] = []
    for item in payload["issues"]:
        if not isinstance(item, dict):
            raise GUIProtocolError("Core returned an invalid validation issue")
        try:
            severity = ValidationSeverity(item.get("severity"))
        except (TypeError, ValueError) as error:
            raise GUIProtocolError("Core returned an unknown validation severity") from error
        code = item.get("code")
        message = item.get("message")
        if not isinstance(code, str) or not code.strip():
            raise GUIProtocolError("Core returned a validation issue without a code")
        if not isinstance(message, str) or not message.strip():
            raise GUIProtocolError("Core returned a validation issue without a message")
        context: dict[str, str | None] = {}
        for field in ("worksite_id", "worker_id", "source"):
            value = item.get(field)
            if value is not None and not isinstance(value, str):
                raise GUIProtocolError(f"Core returned an invalid validation {field}")
            context[field] = value
        issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                worksite_id=context["worksite_id"],
                worker_id=context["worker_id"],
                source=context["source"],
            )
        )
    return ValidationReport(tuple(issues))


def parse_outcome(payload: Any) -> Outcome:
    if not isinstance(payload, dict):
        raise GUIProtocolError("Core returned an invalid outcome")

    try:
        status = OutcomeStatus(payload.get("status"))
    except (TypeError, ValueError) as error:
        raise GUIProtocolError("Core returned an unknown outcome status") from error

    worker_id = payload.get("worker_id")
    missing_ppe = payload.get("missing_ppe")
    if worker_id is not None and not isinstance(worker_id, str):
        raise GUIProtocolError("Core returned an invalid worker ID")
    if missing_ppe is not None and (
        not isinstance(missing_ppe, list) or any(not isinstance(item, str) for item in missing_ppe)
    ):
        raise GUIProtocolError("Core returned an invalid missing-PPE list")

    reason = payload.get("face_failure_reason")
    attempt = payload.get("attempt")
    retry_allowed = payload.get("retry_allowed")
    if reason is not None and not isinstance(reason, str):
        raise GUIProtocolError("Core returned an invalid face failure reason")
    if attempt is not None and (not isinstance(attempt, int) or isinstance(attempt, bool)):
        raise GUIProtocolError("Core returned an invalid face capture attempt")
    if retry_allowed is not None and not isinstance(retry_allowed, bool):
        raise GUIProtocolError("Core returned an invalid retry flag")

    return Outcome(
        status=status,
        worker_id=worker_id,
        missing_ppe=tuple(missing_ppe or ()),
        face_failure_reason=reason,
        attempt=attempt,
        retry_allowed=retry_allowed,
    )
