from __future__ import annotations

import asyncio
import unittest

from signatus_ai.cache import DetectionCache, FrameSnapshot
from signatus_ai.camera import CameraRuntime
from signatus_contracts import CameraState


class FakeTracker:
    def __init__(
        self,
        *,
        open_error: str | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.model_initialized = False
        self.model_loads = 0
        self.camera_opens = 0
        self.camera_releases = 0
        self.inferences = 0
        self.camera_is_open = False
        self.open_error = open_error
        self.run_error = run_error

    def initialize_model(self) -> None:
        if not self.model_initialized:
            self.model_initialized = True
            self.model_loads += 1

    def open_camera(self) -> None:
        self.camera_opens += 1
        if self.open_error is not None:
            raise RuntimeError(self.open_error)
        self.camera_is_open = True

    async def open_camera_async(self) -> None:
        self.open_camera()

    def release_camera(self) -> None:
        if self.camera_is_open:
            self.camera_releases += 1
        self.camera_is_open = False

    async def run(self, stop: asyncio.Event) -> None:
        try:
            if self.run_error is not None:
                raise self.run_error
            while not stop.is_set():
                self.inferences += 1
                await asyncio.sleep(0)
        finally:
            self.release_camera()


class FakePublisher:
    def __init__(self) -> None:
        self.is_open = False
        self.invalidations = 0

    def open(self) -> bool:
        self.is_open = True
        return True

    def invalidate(self) -> None:
        self.invalidations += 1

    def close(self) -> None:
        self.is_open = False


class CameraLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_can_be_ready_while_camera_is_stopped(self) -> None:
        tracker = FakeTracker()
        runtime = CameraRuntime(tracker, DetectionCache(), FakePublisher(), enabled=True)

        runtime.initialize_model()

        self.assertTrue(runtime.model_initialized)
        self.assertEqual(runtime.status().state, CameraState.STOPPED)
        self.assertFalse(runtime.running)

    async def test_stop_releases_camera_stops_inference_and_clears_cache(self) -> None:
        tracker = FakeTracker()
        cache = DetectionCache()
        publisher = FakePublisher()
        runtime = CameraRuntime(tracker, cache, publisher, enabled=True)
        runtime.initialize_model()

        started = await runtime.start()
        await asyncio.sleep(0)
        await cache.store(FrameSnapshot(1.0, (), (), object()))
        before_stop = tracker.inferences
        stopped = await runtime.stop()
        after_stop = tracker.inferences
        await asyncio.sleep(0)

        self.assertEqual(started.state, CameraState.RUNNING)
        self.assertEqual(stopped.state, CameraState.STOPPED)
        self.assertFalse(tracker.camera_is_open)
        self.assertEqual(tracker.camera_releases, 1)
        self.assertEqual(tracker.inferences, after_stop)
        self.assertGreaterEqual(after_stop, before_stop)
        self.assertIsNone(await cache.latest())
        self.assertGreaterEqual(publisher.invalidations, 1)
        self.assertTrue(runtime.model_initialized)

    async def test_restart_reuses_model_and_opens_camera_again(self) -> None:
        tracker = FakeTracker()
        runtime = CameraRuntime(tracker, DetectionCache(), FakePublisher(), enabled=True)
        runtime.initialize_model()

        await runtime.start()
        await asyncio.sleep(0)
        await runtime.stop()
        await runtime.start()
        await asyncio.sleep(0)
        await runtime.stop()

        self.assertEqual(tracker.model_loads, 1)
        self.assertEqual(tracker.camera_opens, 2)
        self.assertEqual(tracker.camera_releases, 2)

    async def test_open_failure_sets_camera_error_without_losing_model(self) -> None:
        tracker = FakeTracker(open_error="device unavailable")
        runtime = CameraRuntime(tracker, DetectionCache(), FakePublisher(), enabled=True)
        runtime.initialize_model()

        status = await runtime.start()

        self.assertEqual(status.state, CameraState.ERROR)
        self.assertEqual(status.error, "device unavailable")
        self.assertTrue(runtime.model_initialized)
        self.assertFalse(runtime.running)
        self.assertFalse(tracker.camera_is_open)

    async def test_inference_component_failure_is_escalated_to_service_health(self) -> None:
        tracker = FakeTracker(run_error=RuntimeError("YOLO inference failed"))
        service_errors: list[str] = []
        runtime = CameraRuntime(
            tracker,
            DetectionCache(),
            FakePublisher(),
            enabled=True,
            on_service_error=service_errors.append,
        )
        runtime.initialize_model()

        await runtime.start()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(runtime.status().state, CameraState.ERROR)
        self.assertEqual(service_errors, ["YOLO inference failed"])


if __name__ == "__main__":
    unittest.main()
