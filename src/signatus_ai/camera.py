"""AI-owned camera lifecycle, independent from AI service health."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Protocol

from signatus_contracts import CameraState, CameraStatus

from .cache import DetectionCache

logger = logging.getLogger(__name__)


class CameraOperationError(RuntimeError):
    """A physical camera open/read failure, not an AI component failure."""


class CameraTracker(Protocol):
    @property
    def model_initialized(self) -> bool: ...

    def initialize_model(self) -> None: ...

    async def open_camera_async(self) -> None: ...

    def release_camera(self) -> None: ...

    async def run(self, stop: asyncio.Event) -> None: ...


class PreviewPublisher(Protocol):
    @property
    def is_open(self) -> bool: ...

    def open(self) -> bool: ...

    def invalidate(self) -> None: ...

    def close(self) -> None: ...


class CameraRuntime:
    """Serialize camera commands while leaving models resident in memory."""

    def __init__(
        self,
        tracker: CameraTracker,
        cache: DetectionCache,
        publisher: PreviewPublisher | None,
        *,
        enabled: bool,
        on_service_error: Callable[[str], None] | None = None,
    ) -> None:
        self._tracker = tracker
        self._cache = cache
        self._publisher = publisher
        self._enabled = enabled
        self._on_service_error = on_service_error
        self._state = CameraState.STOPPED
        self._error: str | None = None
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._operation_lock = asyncio.Lock()

    @property
    def model_initialized(self) -> bool:
        return self._tracker.model_initialized

    @property
    def running(self) -> bool:
        task = self._task
        return self._state is CameraState.RUNNING and task is not None and not task.done()

    def status(self) -> CameraStatus:
        return CameraStatus(state=self._state, error=self._error)

    def initialize_model(self) -> None:
        if not self._enabled:
            raise RuntimeError("AI tracking is disabled")
        self._tracker.initialize_model()

    async def start(self) -> CameraStatus:
        async with self._operation_lock:
            if self._state in {CameraState.STARTING, CameraState.RUNNING}:
                return self.status()

            self._state = CameraState.STARTING
            self._error = None
            await self._cache.clear()
            try:
                if not self._enabled:
                    raise RuntimeError("AI tracking is disabled")
                if not self._tracker.model_initialized:
                    raise RuntimeError("YOLO model is not initialized")
                if self._publisher is not None and not self._publisher.open():
                    raise RuntimeError("Preview shared memory could not be initialized")
                await self._tracker.open_camera_async()
            except Exception as exc:
                await self._cleanup_camera_data()
                self._state = CameraState.ERROR
                self._error = str(exc)
                logger.exception("Camera start failed")
                return self.status()

            stop_event = asyncio.Event()
            self._stop_event = stop_event
            self._state = CameraState.RUNNING
            self._task = asyncio.create_task(
                self._run_tracker(stop_event),
                name="camera-tracker",
            )
            return self.status()

    async def stop(self) -> CameraStatus:
        async with self._operation_lock:
            if self._state is CameraState.STOPPED:
                await self._cleanup_camera_data()
                return self.status()

            self._state = CameraState.STOPPING
            self._error = None
            stop_event = self._stop_event
            task = self._task
            await self._cache.clear()
            if self._publisher is not None:
                self._publisher.invalidate()
            if stop_event is not None:
                stop_event.set()
            if task is not None:
                # OpenCV capture is running in a worker thread. Let the current
                # frame finish so release and SHM teardown cannot race it.
                await asyncio.gather(task, return_exceptions=True)
            await self._cleanup_camera_data()
            self._stop_event = None
            self._task = None
            self._state = CameraState.STOPPED
            return self.status()

    async def shutdown(self) -> None:
        await self.stop()

    async def _run_tracker(self, stop_event: asyncio.Event) -> None:
        failure: Exception | None = None
        try:
            await self._tracker.run(stop_event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = exc
            logger.exception("Camera tracker stopped unexpectedly")
        finally:
            await self._cleanup_camera_data()
            if self._state is CameraState.RUNNING:
                self._state = CameraState.ERROR
                self._error = str(failure or RuntimeError("Camera tracker stopped unexpectedly"))
                if (
                    failure is not None
                    and not isinstance(failure, CameraOperationError)
                    and self._on_service_error is not None
                ):
                    self._on_service_error(str(failure))
            if asyncio.current_task() is self._task:
                self._task = None
                self._stop_event = None

    async def _cleanup_camera_data(self) -> None:
        await self._cache.clear()
        self._tracker.release_camera()
        if self._publisher is not None:
            self._publisher.invalidate()
