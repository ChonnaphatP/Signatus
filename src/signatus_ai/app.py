from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from signatus_contracts import (
    AIServiceState,
    CameraStatus,
    EmbeddingResult,
    FaceEmbeddingStatus,
    PPEResult,
    PPEResultStatus,
)

from .association import (
    AssociationUnresolvedError,
    SinglePersonFrameAssociation,
    UnconfiguredAssociation,
)
from .cache import DetectionCache
from .camera import CameraRuntime
from .embedding import OpenCVYuNetSFaceEmbedder
from .events import EventHub
from .frame_publisher import SharedMemoryFramePublisher
from .settings import AISettings
from .tracker import UltralyticsOpenVINOTracker

logger = logging.getLogger(__name__)

settings = AISettings.from_environment()
cache = DetectionCache()
events = EventHub()
embedder = OpenCVYuNetSFaceEmbedder(
    detector_model_path=settings.face_detector_model_path,
    recognizer_model_path=settings.face_recognizer_model_path,
    score_threshold=settings.face_detector_score_threshold,
    nms_threshold=settings.face_detector_nms_threshold,
    top_k=settings.face_detector_top_k,
)


def _configured_frame_publisher() -> SharedMemoryFramePublisher | None:
    if not settings.frame_shm_enabled:
        return None
    try:
        return SharedMemoryFramePublisher(
            name=settings.frame_shm_name,
            slot_capacity=settings.preview_max_frame_bytes,
        )
    except ValueError:
        logger.exception(
            "Camera preview shared-memory configuration is invalid; "
            "tracking will continue without preview"
        )
        return None


frame_publisher = _configured_frame_publisher()

if settings.ppe_association == "single_person_frame":
    association = SinglePersonFrameAssociation(frozenset({settings.person_class}))
else:
    association = UnconfiguredAssociation()

tracker = UltralyticsOpenVINOTracker(
    model_path=str(settings.model_path),
    camera_source=settings.camera_source,
    person_class=settings.person_class,
    lost_timeout_seconds=settings.track_lost_timeout_seconds,
    cache=cache,
    publish=events.publish,
    frame_publisher=frame_publisher,
)
service_state = AIServiceState.STARTING


class FaceImageEmbeddingRequest(BaseModel):
    face_image: str


def _report_service_error(message: str) -> None:
    global service_state
    service_state = AIServiceState.ERROR
    logger.error("AI component failure during camera operation: %s", message)


camera_runtime = CameraRuntime(
    tracker,
    cache,
    frame_publisher,
    enabled=settings.tracking_enabled,
    on_service_error=_report_service_error,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service_state
    service_state = AIServiceState.STARTING
    try:
        embedder.initialize()
        camera_runtime.initialize_model()
        if frame_publisher is not None and not frame_publisher.open():
            raise RuntimeError("Camera preview shared memory could not be initialized")
        service_state = AIServiceState.READY
        yield
    except Exception:
        service_state = AIServiceState.ERROR
        raise
    finally:
        await camera_runtime.shutdown()
        if frame_publisher is not None:
            frame_publisher.close()
        tracker.close()
        embedder.close()
        service_state = AIServiceState.STOPPED


app = FastAPI(title="Signatus AI Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    snapshot = await cache.latest()
    now = time.time()
    latest_frame_age_seconds = (
        None if snapshot is None else max(0.0, now - snapshot.captured_at)
    )
    preview_timestamp_ns = (
        0 if frame_publisher is None else frame_publisher.last_publish_timestamp_ns
    )
    preview_age_seconds = (
        None
        if preview_timestamp_ns <= 0
        else max(0.0, now - preview_timestamp_ns / 1_000_000_000)
    )
    return {
        "status": "ok" if service_state is AIServiceState.READY else "error",
        "service_state": service_state,
        "camera_state": camera_runtime.status().state,
        "camera_error": camera_runtime.status().error,
        "tracking_enabled": settings.tracking_enabled,
        "tracking_running": camera_runtime.running,
        "yolo_model_initialized": camera_runtime.model_initialized,
        "latest_frame_available": snapshot is not None,
        "latest_frame_age_seconds": latest_frame_age_seconds,
        "model_path": str(settings.model_path),
        "camera_source": settings.camera_source,
        "ppe_association": settings.ppe_association,
        "frame_preview_enabled": settings.frame_shm_enabled,
        "frame_preview_available": frame_publisher is not None and frame_publisher.is_open,
        "frame_preview_published": preview_timestamp_ns > 0,
        "frame_preview_age_seconds": preview_age_seconds,
        "frame_shm_name": settings.frame_shm_name if settings.frame_shm_enabled else None,
        "face_backend": "opencv_yunet_sface_fp32",
        "face_models_available": embedder.model_files_available,
        "face_models_initialized": embedder.models_initialized,
    }


@app.get("/camera", response_model=CameraStatus)
async def camera_status() -> CameraStatus:
    return camera_runtime.status()


@app.post("/commands/camera/start", response_model=CameraStatus)
async def start_camera() -> CameraStatus:
    return await camera_runtime.start()


@app.post("/commands/camera/stop", response_model=CameraStatus)
async def stop_camera() -> CameraStatus:
    return await camera_runtime.stop()


@app.post("/commands/tracks/{track_id}/embedding", response_model=EmbeddingResult)
async def generate_embedding(track_id: int) -> EmbeddingResult:
    if not camera_runtime.running:
        return EmbeddingResult(track_id=track_id, status=FaceEmbeddingStatus.ERROR)
    return await embedder.generate(track_id, await cache.latest())


@app.post("/commands/faces/embedding", response_model=EmbeddingResult)
async def generate_face_image_embedding(
    request: FaceImageEmbeddingRequest,
) -> EmbeddingResult:
    return await embedder.generate_face_image(request.face_image)


@app.post("/commands/tracks/{track_id}/ppe", response_model=PPEResult)
async def get_cached_ppe(track_id: int) -> PPEResult:
    if not camera_runtime.running:
        return PPEResult(track_id=track_id, status=PPEResultStatus.NO_CACHED_FRAME)
    snapshot = await cache.latest()
    if snapshot is None:
        return PPEResult(track_id=track_id, status=PPEResultStatus.NO_CACHED_FRAME)
    if all(person.track_id != track_id for person in snapshot.people):
        return PPEResult(
            track_id=track_id,
            status=PPEResultStatus.TRACK_NOT_FOUND,
            captured_at=snapshot.captured_at,
        )
    try:
        classes = association.classes_for_track(track_id, snapshot)
    except AssociationUnresolvedError:
        return PPEResult(
            track_id=track_id,
            status=PPEResultStatus.ASSOCIATION_UNRESOLVED,
            captured_at=snapshot.captured_at,
        )
    return PPEResult(
        track_id=track_id,
        status=PPEResultStatus.OK,
        detected_classes=list(classes),
        captured_at=snapshot.captured_at,
    )


@app.websocket("/ws/events")
async def tracking_events(socket: WebSocket) -> None:
    await socket.accept()
    queue = events.subscribe()
    event_task = asyncio.create_task(queue.get())
    receive_task = asyncio.create_task(socket.receive())
    try:
        while True:
            done, _pending = await asyncio.wait(
                {event_task, receive_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if receive_task in done:
                message = receive_task.result()
                if message.get("type") == "websocket.disconnect":
                    break
                receive_task = asyncio.create_task(socket.receive())
            if event_task in done:
                event = event_task.result()
                await socket.send_text(event.model_dump_json())
                event_task = asyncio.create_task(queue.get())
    except (RuntimeError, WebSocketDisconnect):
        pass
    finally:
        for task in (event_task, receive_task):
            task.cancel()
        await asyncio.gather(event_task, receive_task, return_exceptions=True)
        events.unsubscribe(queue)
