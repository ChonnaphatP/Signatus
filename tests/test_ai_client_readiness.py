from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from signatus_contracts import CameraState
from signatus_core.ai_client import AIServiceClient, AIServiceClientError
from signatus_core.domain import FaceEmbeddingStatus


class _FakeSocket:
    def __init__(self, message: str) -> None:
        self._message = message

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self._messages()

    async def _messages(self):  # type: ignore[no-untyped-def]
        yield self._message


class _FakeConnection:
    def __init__(self, socket: _FakeSocket) -> None:
        self._socket = socket

    async def __aenter__(self) -> _FakeSocket:
        return self._socket

    async def __aexit__(self, *_args: object) -> None:
        return None


class AIClientReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_embedding_with_wrong_sface_dimensions(self) -> None:
        client = AIServiceClient("http://127.0.0.1:8001", "ws://127.0.0.1:8001/ws/events")
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "track_id": 3,
            "status": "OK",
            "embedding": [1.0, 0.0],
        }
        client._http.post = AsyncMock(return_value=response)
        try:
            result = await client.generate_embedding(3)
        finally:
            await client.close()

        self.assertEqual(result.status, FaceEmbeddingStatus.ERROR)

    async def test_profile_embedding_uses_face_image_route_and_strict_descriptor(self) -> None:
        client = AIServiceClient("http://127.0.0.1:8001", "ws://127.0.0.1:8001/ws/events")
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "track_id": 0,
            "status": "OK",
            "embedding": [1.0] + [0.0] * 127,
        }
        client._http.post = AsyncMock(return_value=response)
        try:
            result = await client.generate_profile_embedding("data:image/png;base64,AA==")
        finally:
            await client.close()

        self.assertEqual(result.status, FaceEmbeddingStatus.OK)
        self.assertEqual(len(result.embedding or ()), 128)
        client._http.post.assert_awaited_once_with(
            "http://127.0.0.1:8001/commands/faces/embedding",
            json={"face_image": "data:image/png;base64,AA=="},
        )

    async def test_profile_embedding_rejects_invalid_ai_descriptor(self) -> None:
        client = AIServiceClient("http://127.0.0.1:8001", "ws://127.0.0.1:8001/ws/events")
        response = Mock()
        response.raise_for_status.return_value = None
        client._http.post = AsyncMock(return_value=response)
        try:
            for descriptor in (
                [1.0, 0.0, 0.0],
                ["1.0"] + [0.0] * 127,
                [True] + [0.0] * 127,
                [0.0] * 128,
            ):
                with self.subTest(first_value=descriptor[0], length=len(descriptor)):
                    response.json.return_value = {
                        "track_id": 0,
                        "status": "OK",
                        "embedding": descriptor,
                    }
                    with self.assertRaises(AIServiceClientError):
                        await client.generate_profile_embedding(
                            "data:image/png;base64,AA=="
                        )
        finally:
            await client.close()

    async def test_runtime_embedding_rejects_string_and_boolean_values(self) -> None:
        client = AIServiceClient("http://127.0.0.1:8001", "ws://127.0.0.1:8001/ws/events")
        response = Mock()
        response.raise_for_status.return_value = None
        client._http.post = AsyncMock(return_value=response)
        try:
            for descriptor in (
                ["1.0"] + [0.0] * 127,
                [True] + [0.0] * 127,
            ):
                with self.subTest(first_value=descriptor[0]):
                    response.json.return_value = {
                        "track_id": 3,
                        "status": "OK",
                        "embedding": descriptor,
                    }
                    result = await client.generate_embedding(3)
                    self.assertEqual(result.status, FaceEmbeddingStatus.ERROR)
        finally:
            await client.close()

    async def test_camera_status_and_commands_use_the_ai_control_routes(self) -> None:
        client = AIServiceClient("http://127.0.0.1:8001", "ws://127.0.0.1:8001/ws/events")
        responses = []
        for state in ("STOPPED", "STARTING", "STOPPING"):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"state": state, "error": None}
            responses.append(response)
        client._http.request = AsyncMock(side_effect=responses)
        try:
            stopped = await client.get_camera_status()
            starting = await client.start_camera()
            stopping = await client.stop_camera()
        finally:
            await client.close()

        self.assertEqual(stopped.state, CameraState.STOPPED)
        self.assertEqual(starting.state, CameraState.STARTING)
        self.assertEqual(stopping.state, CameraState.STOPPING)
        self.assertEqual(
            [call.args for call in client._http.request.await_args_list],
            [
                ("GET", "http://127.0.0.1:8001/camera"),
                ("POST", "http://127.0.0.1:8001/commands/camera/start"),
                ("POST", "http://127.0.0.1:8001/commands/camera/stop"),
            ],
        )

    async def test_invalid_camera_status_is_a_control_error(self) -> None:
        client = AIServiceClient("http://127.0.0.1:8001", "ws://127.0.0.1:8001/ws/events")
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"state": "UNKNOWN", "error": None}
        client._http.request = AsyncMock(return_value=response)
        try:
            with self.assertRaises(AIServiceClientError):
                await client.get_camera_status()
        finally:
            await client.close()

    async def test_reports_connected_only_inside_ai_event_connection(self) -> None:
        client = AIServiceClient("http://127.0.0.1:8001", "ws://127.0.0.1:8001/ws/events")
        stop = __import__("asyncio").Event()
        observed: list[bool] = []

        async def handle(_event: object) -> None:
            observed.append(client.events_connected)
            stop.set()

        connection = _FakeConnection(
            _FakeSocket('{"type":"PERSON_SEEN","track_id":4,"captured_at":1.0}')
        )
        try:
            with patch("signatus_core.ai_client.websockets.connect", return_value=connection):
                await client.listen_forever(handle, stop)
        finally:
            await client.close()

        self.assertEqual(observed, [True])
        self.assertFalse(client.events_connected)


if __name__ == "__main__":
    unittest.main()
