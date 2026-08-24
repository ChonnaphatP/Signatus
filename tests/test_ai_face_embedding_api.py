from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import signatus_ai.app as ai_app
from signatus_ai.embedding import ENROLLMENT_EMBEDDING_TRACK_ID
from signatus_contracts import AIServiceState, EmbeddingResult, FaceEmbeddingStatus


class _EnrollmentEmbedder:
    def __init__(self, result: EmbeddingResult) -> None:
        self.result = result
        self.face_images: list[str] = []

    async def generate_face_image(self, face_image: str) -> EmbeddingResult:
        self.face_images.append(face_image)
        return self.result


class AIFaceEmbeddingAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_works_while_camera_is_stopped(self) -> None:
        result = EmbeddingResult(
            track_id=ENROLLMENT_EMBEDDING_TRACK_ID,
            status=FaceEmbeddingStatus.OK,
            embedding=[1.0] + [0.0] * 127,
        )
        enrollment_embedder = _EnrollmentEmbedder(result)
        request = ai_app.FaceImageEmbeddingRequest(
            face_image="data:image/jpeg;base64,/9j/"
        )

        with (
            patch.object(ai_app, "embedder", enrollment_embedder),
            patch.object(ai_app, "camera_runtime", SimpleNamespace(running=False)),
        ):
            response = await ai_app.generate_face_image_embedding(request)

        self.assertEqual(response.status, FaceEmbeddingStatus.OK)
        self.assertEqual(response.track_id, ENROLLMENT_EMBEDDING_TRACK_ID)
        self.assertEqual(
            enrollment_embedder.face_images,
            ["data:image/jpeg;base64,/9j/"],
        )

    async def test_bad_face_image_does_not_change_ai_service_health(self) -> None:
        request = ai_app.FaceImageEmbeddingRequest(face_image="invalid-image")

        with patch.object(ai_app, "service_state", AIServiceState.READY):
            response = await ai_app.generate_face_image_embedding(request)
            state_after_error = ai_app.service_state

        self.assertEqual(response.status, FaceEmbeddingStatus.ERROR)
        self.assertEqual(response.track_id, ENROLLMENT_EMBEDDING_TRACK_ID)
        self.assertEqual(state_after_error, AIServiceState.READY)

    def test_route_is_exposed_as_a_post_command(self) -> None:
        route = next(
            route
            for route in ai_app.app.routes
            if getattr(route, "path", None) == "/commands/faces/embedding"
        )

        self.assertEqual(route.methods, {"POST"})


if __name__ == "__main__":
    unittest.main()
