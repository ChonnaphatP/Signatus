from __future__ import annotations

import base64
import unittest

from signatus_core.domain import EmbeddingResult, FaceEmbeddingStatus
from signatus_core.enrollment import (
    WorksiteEnrollmentError,
    materialize_worksite_embeddings,
)


def face_image() -> str:
    image = b"\x89PNG\r\n\x1a\n" + b"enrollment-image"
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def embedding(axis: int = 0) -> tuple[float, ...]:
    values = [0.0] * 128
    values[axis] = 1.0
    return tuple(values)


class FakeEmbeddingClient:
    def __init__(self, status: FaceEmbeddingStatus = FaceEmbeddingStatus.OK) -> None:
        self.status = status
        self.images: list[str] = []

    async def generate_profile_embedding(self, image: str) -> EmbeddingResult:
        self.images.append(image)
        return EmbeddingResult(
            track_id=0,
            status=self.status,
            embedding=embedding() if self.status is FaceEmbeddingStatus.OK else None,
        )


class WorksiteEnrollmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_images_become_embeddings_and_are_not_persisted(self) -> None:
        client = FakeEmbeddingClient()
        payload = {
            "worksite_id": "WO-1",
            "name": "Gate",
            "authorized_workers": [
                {"worker_id": "W1", "name": "Worker One", "face_image": face_image()}
            ],
            "required_ppe": [],
        }

        result = await materialize_worksite_embeddings(payload, client)

        worker = result["authorized_workers"][0]
        self.assertEqual(set(worker), {"worker_id", "name", "embedding"})
        self.assertEqual(len(worker["embedding"]), 128)
        self.assertNotIn("face_image", worker)
        self.assertEqual(client.images, [face_image()])

    async def test_existing_worksite_embedding_is_retained_without_ai_call(self) -> None:
        client = FakeEmbeddingClient()
        payload = {
            "worksite_id": "WO-1",
            "name": "Gate",
            "authorized_workers": [
                {"worker_id": "W1", "name": "Worker One", "embedding": list(embedding(2))}
            ],
            "required_ppe": ["helmet"],
        }

        result = await materialize_worksite_embeddings(payload, client)

        self.assertEqual(result["authorized_workers"][0]["embedding"][2], 1.0)
        self.assertEqual(client.images, [])

    async def test_invalid_profile_is_rejected_before_any_ai_call(self) -> None:
        client = FakeEmbeddingClient()
        payload = {
            "worksite_id": "WO-1",
            "name": "Gate",
            "authorized_workers": [
                {"worker_id": "W1", "name": "Worker One", "face_image": "not-data"}
            ],
            "required_ppe": [],
        }

        with self.assertRaises(WorksiteEnrollmentError) as raised:
            await materialize_worksite_embeddings(payload, client)

        self.assertEqual(raised.exception.code, "INVALID_WORKER_PROFILE")
        self.assertEqual(client.images, [])

    async def test_face_outcomes_are_scoped_enrollment_errors(self) -> None:
        payload = {
            "worksite_id": "WO-1",
            "name": "Gate",
            "authorized_workers": [
                {"worker_id": "W1", "name": "Worker One", "face_image": face_image()}
            ],
            "required_ppe": [],
        }
        expectations = {
            FaceEmbeddingStatus.NO_FACE: "WORKER_PROFILE_NO_FACE",
            FaceEmbeddingStatus.MULTIPLE_FACES: "WORKER_PROFILE_MULTIPLE_FACES",
            FaceEmbeddingStatus.LOW_QUALITY: "WORKER_PROFILE_LOW_QUALITY",
            FaceEmbeddingStatus.ERROR: "WORKER_PROFILE_EMBEDDING_ERROR",
        }
        for status, code in expectations.items():
            with self.subTest(status=status):
                with self.assertRaises(WorksiteEnrollmentError) as raised:
                    await materialize_worksite_embeddings(
                        payload,
                        FakeEmbeddingClient(status),
                    )
                self.assertEqual(raised.exception.code, code)

    async def test_worker_cannot_supply_face_image_and_embedding_together(self) -> None:
        payload = {
            "worksite_id": "WO-1",
            "name": "Gate",
            "authorized_workers": [
                {
                    "worker_id": "W1",
                    "name": "Worker One",
                    "face_image": face_image(),
                    "embedding": list(embedding()),
                }
            ],
            "required_ppe": [],
        }

        with self.assertRaises(WorksiteEnrollmentError) as raised:
            await materialize_worksite_embeddings(payload, FakeEmbeddingClient())

        self.assertEqual(raised.exception.code, "AMBIGUOUS_WORKER_ENROLLMENT")


if __name__ == "__main__":
    unittest.main()
