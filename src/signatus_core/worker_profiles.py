from __future__ import annotations

import base64
import binascii
import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from signatus_contracts import MAX_ENROLLMENT_IMAGE_BYTES

SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

_DATA_URI_PATTERN = re.compile(
    r"\Adata:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]*={0,2})\Z"
)
_MAX_ENCODED_IMAGE_CHARACTERS = ((MAX_ENROLLMENT_IMAGE_BYTES + 2) // 3) * 4
_UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


@dataclass(frozen=True, slots=True)
class WorkerProfileIssue:
    """A validation or storage error safe to return through Core APIs."""

    code: str
    message: str
    field: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, str]:
        value = {"code": self.code, "message": self.message}
        if self.field is not None:
            value["field"] = self.field
        if self.source is not None:
            value["source"] = self.source
        return value


@dataclass(frozen=True, slots=True)
class WorkerProfile:
    """A reusable v1 worker identity and source face image."""

    worker_id: str
    name: str
    face_image: str

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "face_image": self.face_image,
        }


@dataclass(frozen=True, slots=True)
class WorkerProfileValidationResult:
    profile: WorkerProfile | None
    errors: tuple[WorkerProfileIssue, ...] = ()
    source: str | None = None

    @property
    def valid(self) -> bool:
        return self.profile is not None and not self.errors

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "valid": self.valid,
            "errors": [error.to_dict() for error in self.errors],
        }
        if self.source is not None:
            value["source"] = self.source
        if self.profile is not None:
            value["profile"] = self.profile.to_dict()
        return value


@dataclass(frozen=True, slots=True)
class WorkerProfileRecord:
    """One stored profile, including invalid files that remain inspectable."""

    source: str
    profile: WorkerProfile | None
    errors: tuple[WorkerProfileIssue, ...]
    raw_text: str | None

    @property
    def valid(self) -> bool:
        return self.profile is not None and not self.errors

    def to_dict(self, *, include_raw: bool = False) -> dict[str, object]:
        value: dict[str, object] = {
            "source": self.source,
            "valid": self.valid,
            "errors": [error.to_dict() for error in self.errors],
        }
        if self.profile is not None:
            value["profile"] = self.profile.to_dict()
        if include_raw:
            value["raw_text"] = self.raw_text
        return value


@dataclass(frozen=True, slots=True)
class WorkerProfileSaveResult:
    success: bool
    profile: WorkerProfile | None = None
    source: str | None = None
    path: Path | None = None
    errors: tuple[WorkerProfileIssue, ...] = ()

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "success": self.success,
            "errors": [error.to_dict() for error in self.errors],
        }
        if self.profile is not None:
            value["profile"] = self.profile.to_dict()
        if self.source is not None:
            value["source"] = self.source
        if self.path is not None:
            value["path"] = str(self.path)
        return value


def validate_worker_profile(
    payload: object, *, source: str | None = None
) -> WorkerProfileValidationResult:
    """Validate a decoded v1 Worker Profile without repairing malformed data."""

    if not isinstance(payload, Mapping):
        return _validation_failure(
            "INVALID_WORKER_PROFILE",
            "Worker Profile must be a JSON object.",
            source=source,
        )

    errors: list[WorkerProfileIssue] = []
    expected_fields = {"worker_id", "name", "face_image"}
    unexpected_fields = sorted(str(field) for field in payload if field not in expected_fields)
    if unexpected_fields:
        errors.append(
            _issue(
                "UNEXPECTED_PROFILE_FIELDS",
                "Worker Profile contains unsupported fields: "
                + ", ".join(unexpected_fields),
                source=source,
            )
        )

    worker_id = _nonempty_string(payload.get("worker_id"))
    if worker_id is None:
        errors.append(
            _issue(
                "INVALID_WORKER_ID",
                "Worker Profile requires a non-empty string worker_id.",
                field="worker_id",
                source=source,
            )
        )

    name = _nonempty_string(payload.get("name"))
    if name is None:
        errors.append(
            _issue(
                "INVALID_WORKER_NAME",
                "Worker Profile requires a non-empty string name.",
                field="name",
                source=source,
            )
        )

    face_image = payload.get("face_image")
    if not isinstance(face_image, str):
        errors.append(
            _issue(
                "INVALID_FACE_IMAGE",
                "Worker Profile requires a complete base64 image data URI.",
                field="face_image",
                source=source,
            )
        )
        validated_face_image = None
    else:
        try:
            decode_face_image_data_uri(face_image)
        except ValueError as error:
            errors.append(
                _issue(
                    "INVALID_FACE_IMAGE",
                    str(error),
                    field="face_image",
                    source=source,
                )
            )
            validated_face_image = None
        else:
            validated_face_image = face_image

    if errors or worker_id is None or name is None or validated_face_image is None:
        return WorkerProfileValidationResult(
            profile=None,
            errors=tuple(errors),
            source=source,
        )

    return WorkerProfileValidationResult(
        profile=WorkerProfile(
            worker_id=worker_id,
            name=name,
            face_image=validated_face_image,
        ),
        source=source,
    )


def parse_worker_profile_json(
    raw_text: str, *, source: str | None = None
) -> WorkerProfileValidationResult:
    """Parse and validate a Worker Profile JSON document."""

    try:
        payload = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as error:
        return _validation_failure(
            "INVALID_WORKER_PROFILE_JSON",
            f"Worker Profile cannot be read as JSON: {error}",
            source=source,
        )
    return validate_worker_profile(payload, source=source)


def load_worker_profile_file(path: Path) -> WorkerProfileValidationResult:
    """Read and validate a profile selected from any operator-accessible location."""

    source = path.name
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return _validation_failure(
            "UNREADABLE_WORKER_PROFILE",
            f"Worker Profile cannot be read: {error}",
            source=source,
        )
    return parse_worker_profile_json(raw_text, source=source)


def encode_face_image(
    image_bytes: bytes,
    *,
    mime_type: str | None = None,
) -> str:
    """Encode supported image bytes as a complete, MIME-preserving data URI.

    The file signature is authoritative. A supplied MIME type is accepted only
    when supported and consistent with the detected image type.
    """

    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("Face image bytes must be non-empty bytes.")

    detected_mime_type = detect_image_mime_type(image_bytes)
    if detected_mime_type is None:
        raise ValueError("Face image is not a supported JPEG, PNG, or WebP image.")
    if mime_type is not None:
        normalized_mime_type = mime_type.strip().lower()
        if normalized_mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported face image MIME type: {mime_type!r}.")
        if normalized_mime_type != detected_mime_type:
            raise ValueError(
                "Supplied face image MIME type does not match the image content."
            )

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{detected_mime_type};base64,{encoded}"


def decode_face_image_data_uri(data_uri: str) -> tuple[str, bytes]:
    """Validate and decode a supported complete image data URI."""

    if len(data_uri) > _MAX_ENCODED_IMAGE_CHARACTERS + 64:
        raise ValueError(
            f"Face image must not exceed {MAX_ENROLLMENT_IMAGE_BYTES} decoded bytes."
        )
    match = _DATA_URI_PATTERN.fullmatch(data_uri)
    if match is None:
        raise ValueError(
            "Face image must be a complete JPEG, PNG, or WebP base64 data URI."
        )
    mime_type, encoded = match.groups()
    if not encoded:
        raise ValueError("Face image data URI contains no image data.")
    if len(encoded) > _MAX_ENCODED_IMAGE_CHARACTERS:
        raise ValueError(
            f"Face image must not exceed {MAX_ENROLLMENT_IMAGE_BYTES} decoded bytes."
        )
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Face image data URI contains invalid base64 data.") from error
    if len(image_bytes) > MAX_ENROLLMENT_IMAGE_BYTES:
        raise ValueError(
            f"Face image must not exceed {MAX_ENROLLMENT_IMAGE_BYTES} decoded bytes."
        )

    detected_mime_type = detect_image_mime_type(image_bytes)
    if detected_mime_type is None:
        raise ValueError("Face image data is not a supported JPEG, PNG, or WebP image.")
    if detected_mime_type != mime_type:
        raise ValueError("Face image MIME type does not match the image content.")
    return mime_type, image_bytes


def detect_image_mime_type(image_bytes: bytes) -> str | None:
    """Detect the supported image type from a conservative file signature."""

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


def sanitize_profile_filename(worker_id: str) -> str:
    """Derive a safe JSON filename without changing the internal worker ID."""

    sanitized = _UNSAFE_FILENAME_CHARACTERS.sub("_", worker_id.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip(" ._-")
    if not sanitized:
        sanitized = "worker_profile"
    if sanitized.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_FILENAMES:
        sanitized = f"_{sanitized}"
    sanitized = sanitized[:120].rstrip(" .") or "worker_profile"
    return f"{sanitized}.json"


class WorkerProfileRepository:
    """Filesystem storage for reusable Worker Profiles.

    The directory is injected by the caller. Profiles are never used as a live
    dependency by a Wo.No. The Wo.No. Create/Edit workflow asks AI to derive a
    fresh SFace embedding from the stored face image and copies that descriptor
    into the Wo.No. document.
    """

    def __init__(self, directory: Path):
        self._directory = directory
        self._lock = threading.RLock()

    @property
    def directory(self) -> Path:
        return self._directory

    def list_records(self) -> tuple[WorkerProfileRecord, ...]:
        try:
            paths = sorted(self._directory.glob("*.json"))
        except OSError:
            return ()
        return tuple(self.load(path.name) for path in paths)

    def load(self, source: str) -> WorkerProfileRecord:
        path, path_issue = self._resolve_source(source)
        if path_issue is not None:
            return WorkerProfileRecord(source, None, (path_issue,), None)
        assert path is not None
        try:
            raw_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            return WorkerProfileRecord(
                source,
                None,
                (
                    _issue(
                        "UNREADABLE_WORKER_PROFILE",
                        f"Worker Profile cannot be read: {error}",
                        source=source,
                    ),
                ),
                None,
            )
        result = parse_worker_profile_json(raw_text, source=source)
        return WorkerProfileRecord(source, result.profile, result.errors, raw_text)

    def read_raw(self, source: str) -> WorkerProfileRecord:
        """Return validated metadata and raw JSON for a safe stored basename."""

        return self.load(source)

    def find_sources_by_worker_id(
        self, worker_id: str, *, exclude_source: str | None = None
    ) -> tuple[str, ...]:
        """Find duplicates by internal JSON identity, never by filename."""

        matches: list[str] = []
        try:
            paths = sorted(self._directory.glob("*.json"))
        except OSError:
            return ()
        for path in paths:
            if path.name == exclude_source:
                continue
            safe_path, issue = self._resolve_source(path.name)
            if issue is not None or safe_path is None:
                continue
            try:
                payload = json.loads(safe_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping) and payload.get("worker_id") == worker_id:
                matches.append(path.name)
        return tuple(matches)

    def create(
        self,
        payload: object,
        *,
        filename: str | None = None,
    ) -> WorkerProfileSaveResult:
        result = validate_worker_profile(payload)
        if not result.valid:
            return WorkerProfileSaveResult(False, errors=result.errors)
        assert result.profile is not None

        with self._lock:
            duplicates = self.find_sources_by_worker_id(result.profile.worker_id)
            if duplicates:
                return WorkerProfileSaveResult(
                    False,
                    profile=result.profile,
                    errors=(
                        _issue(
                            "DUPLICATE_WORKER_ID",
                            f"Worker ID {result.profile.worker_id!r} already exists in "
                            + ", ".join(duplicates)
                            + ".",
                            field="worker_id",
                        ),
                    ),
                )

            try:
                self._directory.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                return _storage_failure("WORKER_PROFILE_DIRECTORY_ERROR", error)

            requested_name = (
                sanitize_profile_filename(result.profile.worker_id)
                if filename is None
                else sanitize_profile_filename(Path(filename).stem)
            )
            try:
                path = self._next_available_path(requested_name)
            except OSError as error:
                return _storage_failure("WORKER_PROFILE_FILENAME_ERROR", error)
            write_issue = self._atomic_write(path, result.profile)
            if write_issue is not None:
                return WorkerProfileSaveResult(
                    False,
                    profile=result.profile,
                    errors=(write_issue,),
                )
            return WorkerProfileSaveResult(
                True,
                profile=result.profile,
                source=path.name,
                path=path,
            )

    def edit(self, source: str, payload: object) -> WorkerProfileSaveResult:
        """Replace a stored profile while preserving its original worker ID."""

        with self._lock:
            existing = self.load(source)
            if not existing.valid or existing.profile is None:
                return WorkerProfileSaveResult(False, source=source, errors=existing.errors)

            result = validate_worker_profile(payload, source=source)
            if not result.valid or result.profile is None:
                return WorkerProfileSaveResult(False, source=source, errors=result.errors)
            if result.profile.worker_id != existing.profile.worker_id:
                return WorkerProfileSaveResult(
                    False,
                    profile=result.profile,
                    source=source,
                    errors=(
                        _issue(
                            "IMMUTABLE_WORKER_ID",
                            "Worker ID cannot be changed while editing a Worker Profile.",
                            field="worker_id",
                            source=source,
                        ),
                    ),
                )

            duplicates = self.find_sources_by_worker_id(
                result.profile.worker_id, exclude_source=source
            )
            if duplicates:
                return WorkerProfileSaveResult(
                    False,
                    profile=result.profile,
                    source=source,
                    errors=(
                        _issue(
                            "DUPLICATE_WORKER_ID",
                            f"Worker ID {result.profile.worker_id!r} also exists in "
                            + ", ".join(duplicates)
                            + ".",
                            field="worker_id",
                            source=source,
                        ),
                    ),
                )

            path, path_issue = self._resolve_source(source)
            if path_issue is not None or path is None:
                return WorkerProfileSaveResult(
                    False,
                    profile=result.profile,
                    source=source,
                    errors=(path_issue,) if path_issue is not None else (),
                )
            write_issue = self._atomic_write(path, result.profile)
            if write_issue is not None:
                return WorkerProfileSaveResult(
                    False,
                    profile=result.profile,
                    source=source,
                    errors=(write_issue,),
                )
            return WorkerProfileSaveResult(
                True,
                profile=result.profile,
                source=source,
                path=path,
            )

    def _resolve_source(
        self, source: str
    ) -> tuple[Path | None, WorkerProfileIssue | None]:
        if (
            not isinstance(source, str)
            or not source
            or source in {".", ".."}
            or "/" in source
            or "\\" in source
            or "\x00" in source
            or Path(source).name != source
            or Path(source).suffix.lower() != ".json"
        ):
            return None, _issue(
                "INVALID_PROFILE_SOURCE",
                "Worker Profile source must be a safe JSON filename.",
                source=str(source),
            )

        root = self._directory.resolve()
        candidate = self._directory / source
        try:
            resolved = candidate.resolve()
        except OSError as error:
            return None, _issue(
                "INVALID_PROFILE_SOURCE",
                f"Worker Profile source cannot be resolved: {error}",
                source=source,
            )
        if resolved.parent != root:
            return None, _issue(
                "INVALID_PROFILE_SOURCE",
                "Worker Profile source resolves outside the configured directory.",
                source=source,
            )
        return candidate, None

    def _next_available_path(self, requested_name: str) -> Path:
        path = self._directory / requested_name
        if not path.exists():
            return path
        stem = path.stem
        for index in range(2, 10_000):
            candidate = self._directory / f"{stem}_{index}.json"
            if not candidate.exists():
                return candidate
        raise OSError("Unable to allocate a unique Worker Profile filename.")

    def _atomic_write(
        self, path: Path, profile: WorkerProfile
    ) -> WorkerProfileIssue | None:
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=self._directory,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
                json.dump(
                    profile.to_dict(),
                    output,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            return None
        except (OSError, TypeError, ValueError) as error:
            return _issue(
                "WORKER_PROFILE_WRITE_FAILED",
                f"Worker Profile could not be saved atomically: {error}",
                source=path.name,
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _issue(
    code: str,
    message: str,
    *,
    field: str | None = None,
    source: str | None = None,
) -> WorkerProfileIssue:
    return WorkerProfileIssue(code=code, message=message, field=field, source=source)


def _validation_failure(
    code: str,
    message: str,
    *,
    source: str | None = None,
) -> WorkerProfileValidationResult:
    return WorkerProfileValidationResult(
        profile=None,
        errors=(_issue(code, message, source=source),),
        source=source,
    )


def _storage_failure(code: str, error: OSError) -> WorkerProfileSaveResult:
    return WorkerProfileSaveResult(
        False,
        errors=(_issue(code, f"Worker Profile storage is unavailable: {error}"),),
    )
