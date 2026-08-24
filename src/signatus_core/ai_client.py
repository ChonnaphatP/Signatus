from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable

import httpx
import websockets
from websockets.exceptions import WebSocketException

from signatus_contracts import SFACE_DESCRIPTOR_DIMENSIONS, CameraState, CameraStatus

from .domain import (
    AIEvent,
    AIEventType,
    EmbeddingResult,
    FaceEmbeddingStatus,
    PPEResult,
    PPEResultStatus,
)


class AIServiceClientError(RuntimeError):
    """Raised when an operational AI control request cannot be completed."""


class AIServiceClient:
    def __init__(self, base_url: str, events_url: str):
        self._base_url = base_url.rstrip("/")
        self._events_url = events_url
        self._http = httpx.AsyncClient(timeout=5.0)
        self._events_connected = False

    @property
    def events_connected(self) -> bool:
        return self._events_connected

    async def close(self) -> None:
        self._events_connected = False
        await self._http.aclose()

    async def generate_embedding(self, track_id: int) -> EmbeddingResult:
        try:
            response = await self._http.post(
                f"{self._base_url}/commands/tracks/{track_id}/embedding"
            )
            response.raise_for_status()
            raw = response.json()
            status = FaceEmbeddingStatus(raw["status"])
            values = _strict_sface_descriptor(raw.get("embedding"))
            if (status is FaceEmbeddingStatus.OK) != (values is not None):
                return EmbeddingResult(track_id, FaceEmbeddingStatus.ERROR)
            return EmbeddingResult(
                track_id=track_id,
                status=status,
                embedding=values,
            )
        except (
            httpx.HTTPError,
            AttributeError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ):
            return EmbeddingResult(track_id, FaceEmbeddingStatus.ERROR)

    async def generate_profile_embedding(self, face_image: str) -> EmbeddingResult:
        """Ask AI to embed one stored enrollment image without using the camera."""

        try:
            response = await self._http.post(
                f"{self._base_url}/commands/faces/embedding",
                json={"face_image": face_image},
            )
            response.raise_for_status()
            raw = response.json()
            status = FaceEmbeddingStatus(raw["status"])
            values = _strict_sface_descriptor(raw.get("embedding"))
            if (status is FaceEmbeddingStatus.OK) != (values is not None):
                raise ValueError("AI returned an invalid SFace descriptor")
            return EmbeddingResult(
                track_id=0,
                status=status,
                embedding=values,
            )
        except (
            httpx.HTTPError,
            AttributeError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            raise AIServiceClientError("AI face-image embedding request failed") from exc

    async def get_cached_ppe(self, track_id: int) -> PPEResult:
        try:
            response = await self._http.post(f"{self._base_url}/commands/tracks/{track_id}/ppe")
            response.raise_for_status()
            raw = response.json()
            return PPEResult(
                track_id=track_id,
                status=PPEResultStatus(raw["status"]),
                detected_classes=tuple(raw.get("detected_classes", ())),
                captured_at=raw.get("captured_at"),
            )
        except (httpx.HTTPError, AttributeError, KeyError, TypeError, ValueError):
            return PPEResult(track_id, PPEResultStatus.ERROR)

    async def get_camera_status(self) -> CameraStatus:
        return await self._camera_request("GET", "/camera")

    async def start_camera(self) -> CameraStatus:
        return await self._camera_request("POST", "/commands/camera/start")

    async def stop_camera(self) -> CameraStatus:
        return await self._camera_request("POST", "/commands/camera/stop")

    async def _camera_request(self, method: str, path: str) -> CameraStatus:
        try:
            response = await self._http.request(method, f"{self._base_url}{path}")
            response.raise_for_status()
            raw = response.json()
            if not isinstance(raw, dict):
                raise TypeError("camera response is not an object")
            state = CameraState(raw["state"])
            error = raw.get("error")
            if error is not None and not isinstance(error, str):
                raise TypeError("camera error is not text")
            return CameraStatus(state=state, error=error)
        except (httpx.HTTPError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise AIServiceClientError(
                f"AI camera request {method} {path} failed"
            ) from exc

    async def listen_forever(
        self,
        handler: Callable[[AIEvent], Awaitable[None]],
        stop: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                async with websockets.connect(self._events_url) as socket:
                    self._events_connected = True
                    try:
                        async for message in socket:
                            try:
                                event = self._parse_event(message)
                            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                                continue
                            await handler(event)
                            if stop.is_set():
                                return
                    finally:
                        self._events_connected = False
            except asyncio.CancelledError:
                raise
            except (OSError, TimeoutError, WebSocketException):
                self._events_connected = False
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1.0)
                except TimeoutError:
                    pass
        self._events_connected = False

    @staticmethod
    def _parse_event(message: str | bytes) -> AIEvent:
        raw = json.loads(message)
        return AIEvent(
            type=AIEventType(raw["type"]),
            track_id=int(raw["track_id"]),
            captured_at=float(raw["captured_at"]),
        )


def _strict_sface_descriptor(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != SFACE_DESCRIPTOR_DIMENSIONS:
        raise ValueError("AI returned an invalid SFace descriptor shape")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in value
    ):
        raise TypeError("AI returned a non-numeric SFace descriptor value")
    values = tuple(float(item) for item in value)
    norm = math.hypot(*values)
    if any(not math.isfinite(item) for item in values) or not math.isfinite(norm) or norm == 0.0:
        raise ValueError("AI returned an unusable SFace descriptor")
    return values
