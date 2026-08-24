from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

from .domain import EmbeddingResult, FaceEmbeddingStatus
from .worker_profiles import WorkerProfile, validate_worker_profile


class ProfileEmbeddingClient(Protocol):
    async def generate_profile_embedding(self, face_image: str) -> EmbeddingResult: ...


class WorksiteEnrollmentError(ValueError):
    """A scoped Worker Profile error that must not terminate Core."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        worker_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.worker_id = worker_id

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "worker_id": self.worker_id,
        }


async def materialize_worksite_embeddings(
    payload: object,
    client: ProfileEmbeddingClient,
) -> object:
    """Replace transient face images with AI-generated SFace descriptors.

    Existing compact Wo.No. workers that already contain an embedding remain
    unchanged. This supports editing an existing file without re-enrolling its
    workers, while newly added Worker Profiles are embedded by AI at save time.
    """

    if not isinstance(payload, Mapping):
        return payload
    raw_workers = payload.get("authorized_workers")
    if not isinstance(raw_workers, list):
        return dict(payload)

    prepared: list[tuple[str, object]] = []
    for index, raw_worker in enumerate(raw_workers):
        if not isinstance(raw_worker, Mapping):
            prepared.append(("unchanged", raw_worker))
            continue
        has_face_image = raw_worker.get("face_image") is not None
        has_embedding = raw_worker.get("embedding") is not None
        worker_id_value = raw_worker.get("worker_id")
        worker_id = worker_id_value if isinstance(worker_id_value, str) else None
        if has_face_image and has_embedding:
            raise WorksiteEnrollmentError(
                "AMBIGUOUS_WORKER_ENROLLMENT",
                f"Worker record {index + 1} contains both face_image and embedding.",
                worker_id=worker_id,
            )
        if not has_face_image:
            prepared.append(("unchanged", raw_worker))
            continue

        validation = validate_worker_profile(raw_worker, source="Wo.No. Generator")
        if not validation.valid or validation.profile is None:
            message = (
                validation.errors[0].message
                if validation.errors
                else f"Worker record {index + 1} is not a valid Worker Profile."
            )
            raise WorksiteEnrollmentError(
                "INVALID_WORKER_PROFILE",
                message,
                worker_id=worker_id,
            )
        prepared.append(("profile", validation.profile))

    materialized: list[object] = []
    for kind, value in prepared:
        if kind == "unchanged":
            if isinstance(value, Mapping):
                materialized.append(
                    {
                        "worker_id": value.get("worker_id"),
                        "name": value.get("name"),
                        "embedding": value.get("embedding"),
                    }
                )
            else:
                materialized.append(value)
            continue

        profile = cast(WorkerProfile, value)
        result = await client.generate_profile_embedding(profile.face_image)
        if result.status is not FaceEmbeddingStatus.OK or result.embedding is None:
            code, reason = _embedding_failure(result.status)
            raise WorksiteEnrollmentError(
                code,
                f"Worker {profile.worker_id}: {reason}",
                worker_id=profile.worker_id,
            )
        materialized.append(
            {
                "worker_id": profile.worker_id,
                "name": profile.name,
                "embedding": list(result.embedding),
            }
        )

    return {
        "worksite_id": payload.get("worksite_id"),
        "name": payload.get("name"),
        "authorized_workers": materialized,
        "required_ppe": payload.get("required_ppe"),
    }


def _embedding_failure(status: FaceEmbeddingStatus) -> tuple[str, str]:
    return {
        FaceEmbeddingStatus.NO_FACE: (
            "WORKER_PROFILE_NO_FACE",
            "the profile image contains no detectable face.",
        ),
        FaceEmbeddingStatus.MULTIPLE_FACES: (
            "WORKER_PROFILE_MULTIPLE_FACES",
            "the profile image must contain exactly one face.",
        ),
        FaceEmbeddingStatus.LOW_QUALITY: (
            "WORKER_PROFILE_LOW_QUALITY",
            "the profile image did not produce a usable SFace descriptor.",
        ),
        FaceEmbeddingStatus.ERROR: (
            "WORKER_PROFILE_EMBEDDING_ERROR",
            "AI could not process the profile image.",
        ),
        FaceEmbeddingStatus.OK: (
            "WORKER_PROFILE_EMBEDDING_ERROR",
            "AI returned no SFace descriptor.",
        ),
    }[status]
