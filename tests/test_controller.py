import unittest

from signatus_contracts import CameraState, CameraStatus
from signatus_core.ai_client import AIServiceClientError
from signatus_core.controller import CameraCommandError, CoreController
from signatus_core.domain import (
    AIEvent,
    AIEventType,
    AuthorizedWorker,
    EmbeddingResult,
    FaceEmbeddingStatus,
    OutcomeStatus,
    PPEClassRule,
    PPEResult,
    PPEResultStatus,
    Worksite,
)


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class FakeAI:
    def __init__(
        self,
        embeddings,
        ppe_results=(),
        *,
        camera_status=None,
        start_result=None,
        stop_result=None,
    ):
        self.embeddings = list(embeddings)
        self.ppe_results = list(ppe_results)
        self.embedding_calls = 0
        self.ppe_calls = 0
        self.camera_status = camera_status or CameraStatus(state=CameraState.STOPPED)
        self.start_result = start_result or CameraStatus(state=CameraState.STARTING)
        self.stop_result = stop_result or CameraStatus(state=CameraState.STOPPING)
        self.camera_status_calls = 0
        self.start_calls = 0
        self.stop_calls = 0

    async def generate_embedding(self, track_id):
        self.embedding_calls += 1
        return self.embeddings.pop(0)

    async def get_cached_ppe(self, track_id):
        self.ppe_calls += 1
        return self.ppe_results.pop(0)

    async def get_camera_status(self):
        self.camera_status_calls += 1
        return self.camera_status

    async def start_camera(self):
        self.start_calls += 1
        return self.start_result

    async def stop_camera(self):
        self.stop_calls += 1
        return self.stop_result


class FakeSignals:
    def __init__(self):
        self.items = []

    async def emit(self, signal):
        self.items.append(signal)


WORKSITE = Worksite(
    worksite_id="WO-014",
    name="Cold steel work",
    authorized_workers=(AuthorizedWorker("EMP0017", (1.0, 0.0, 0.0)),),
    required_ppe=("helmet", "gloves"),
)
POLICY = {
    "helmet": PPEClassRule(frozenset({"helmet"}), frozenset({"no_helmet"})),
    "gloves": PPEClassRule(frozenset({"gloves"}), frozenset({"no_gloves"})),
}
RUNNING_CAMERA = CameraStatus(state=CameraState.RUNNING)


def person_seen(track_id=3):
    return AIEvent(AIEventType.PERSON_SEEN, track_id, 1000.0)


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_flow(self):
        ai = FakeAI(
            [EmbeddingResult(3, FaceEmbeddingStatus.OK, (1.0, 0.0, 0.0))],
            [PPEResult(3, PPEResultStatus.OK, ("helmet", "gloves"), 1000.0)],
        )
        signals = FakeSignals()
        controller = CoreController(
            ai, signals, POLICY, initial_camera_status=RUNNING_CAMERA
        )
        controller.select_worksite(WORKSITE)

        await controller.handle_event(person_seen())

        self.assertEqual(signals.items[-1].status, OutcomeStatus.AUTHORIZED)
        self.assertEqual(signals.items[-1].worker_id, "EMP0017")
        await controller.handle_event(person_seen())
        self.assertEqual(ai.embedding_calls, 1)

    async def test_ppe_violation_flow(self):
        ai = FakeAI(
            [EmbeddingResult(3, FaceEmbeddingStatus.OK, (1.0, 0.0, 0.0))],
            [PPEResult(3, PPEResultStatus.OK, ("helmet", "no_gloves"), 1000.0)],
        )
        signals = FakeSignals()
        controller = CoreController(
            ai, signals, POLICY, initial_camera_status=RUNNING_CAMERA
        )
        controller.select_worksite(WORKSITE)

        await controller.handle_event(person_seen())

        signal = signals.items[-1]
        self.assertEqual(signal.status, OutcomeStatus.PPE_VIOLATION)
        self.assertEqual(signal.worker_id, "EMP0017")
        self.assertEqual(signal.missing_ppe, ("gloves",))

    async def test_full_absence_class_marks_every_required_item_missing(self):
        ai = FakeAI(
            [EmbeddingResult(3, FaceEmbeddingStatus.OK, (1.0, 0.0, 0.0))],
            [PPEResult(3, PPEResultStatus.OK, ("helmet", "gloves", "none"), 1000.0)],
        )
        signals = FakeSignals()
        controller = CoreController(
            ai, signals, POLICY, initial_camera_status=RUNNING_CAMERA
        )
        controller.select_worksite(WORKSITE)

        await controller.handle_event(person_seen())

        signal = signals.items[-1]
        self.assertEqual(signal.status, OutcomeStatus.PPE_VIOLATION)
        self.assertEqual(signal.worker_id, "EMP0017")
        self.assertEqual(signal.missing_ppe, WORKSITE.required_ppe)

    async def test_unresolved_association_fails_closed_as_no_detections(self):
        ai = FakeAI(
            [EmbeddingResult(3, FaceEmbeddingStatus.OK, (1.0, 0.0, 0.0))],
            [PPEResult(3, PPEResultStatus.ASSOCIATION_UNRESOLVED, (), 1000.0)],
        )
        signals = FakeSignals()
        controller = CoreController(
            ai, signals, POLICY, initial_camera_status=RUNNING_CAMERA
        )
        controller.select_worksite(WORKSITE)

        await controller.handle_event(person_seen())

        signal = signals.items[-1]
        self.assertEqual(signal.status, OutcomeStatus.PPE_VIOLATION)
        self.assertEqual(signal.missing_ppe, WORKSITE.required_ppe)

    async def test_unknown_embedding_is_unauthorized_without_ppe_call(self):
        ai = FakeAI([EmbeddingResult(3, FaceEmbeddingStatus.OK, (0.0, 1.0, 0.0))])
        signals = FakeSignals()
        controller = CoreController(
            ai,
            signals,
            POLICY,
            face_match_min_cosine_similarity=0.35,
            initial_camera_status=RUNNING_CAMERA,
        )
        controller.select_worksite(WORKSITE)

        await controller.handle_event(person_seen())

        self.assertEqual(signals.items[-1].status, OutcomeStatus.UNAUTHORIZED)
        self.assertEqual(ai.ppe_calls, 0)

    async def test_face_capture_retries_three_times_then_waits_for_track_lost(self):
        failures = [
            EmbeddingResult(3, FaceEmbeddingStatus.NO_FACE),
            EmbeddingResult(3, FaceEmbeddingStatus.LOW_QUALITY),
            EmbeddingResult(3, FaceEmbeddingStatus.NO_FACE),
            EmbeddingResult(3, FaceEmbeddingStatus.NO_FACE),
        ]
        ai = FakeAI(failures)
        signals = FakeSignals()
        clock = FakeClock(0.0)
        controller = CoreController(
            ai,
            signals,
            POLICY,
            clock=clock,
            initial_camera_status=RUNNING_CAMERA,
        )
        controller.select_worksite(WORKSITE)

        await controller.handle_event(person_seen())
        self.assertEqual(signals.items[-1].attempt, 1)
        self.assertTrue(signals.items[-1].retry_allowed)

        clock.value = 0.9
        await controller.handle_event(person_seen())
        self.assertEqual(ai.embedding_calls, 1)

        clock.value = 1.0
        await controller.handle_event(person_seen())
        clock.value = 2.0
        await controller.handle_event(person_seen())
        self.assertEqual(signals.items[-1].attempt, 3)
        self.assertFalse(signals.items[-1].retry_allowed)

        clock.value = 10.0
        await controller.handle_event(person_seen())
        self.assertEqual(ai.embedding_calls, 3)

        await controller.handle_event(AIEvent(AIEventType.TRACK_LOST, 3, 1010.0))
        await controller.handle_event(person_seen())
        self.assertEqual(ai.embedding_calls, 4)
        self.assertEqual(signals.items[-1].attempt, 1)

    async def test_camera_stopped_prevents_authorization(self):
        ai = FakeAI([EmbeddingResult(3, FaceEmbeddingStatus.OK, (1.0, 0.0, 0.0))])
        signals = FakeSignals()
        controller = CoreController(ai, signals, POLICY)
        controller.select_worksite(WORKSITE)

        await controller.handle_event(person_seen())

        self.assertEqual(ai.embedding_calls, 0)
        self.assertEqual(signals.items, [])
        self.assertEqual(controller.camera_status.state, CameraState.STOPPED)

    async def test_validated_start_and_stop_are_forwarded_and_clear_tracks(self):
        ai = FakeAI(
            [],
            start_result=CameraStatus(state=CameraState.RUNNING),
            stop_result=CameraStatus(state=CameraState.STOPPED),
        )
        signals = FakeSignals()
        controller = CoreController(ai, signals, POLICY)

        started = await controller.start_camera()
        controller._track_guard.observe(9, 1.0)
        stopped = await controller.stop_camera()

        self.assertEqual(started.state, CameraState.RUNNING)
        self.assertEqual(stopped.state, CameraState.STOPPED)
        self.assertEqual((ai.start_calls, ai.stop_calls), (1, 1))
        self.assertEqual(controller._track_guard.active_track_ids(), ())

    async def test_invalid_camera_transition_is_rejected_before_ai_call(self):
        ai = FakeAI([])
        controller = CoreController(
            ai,
            FakeSignals(),
            POLICY,
            initial_camera_status=RUNNING_CAMERA,
        )

        with self.assertRaisesRegex(RuntimeError, "cannot start camera while it is RUNNING"):
            await controller.start_camera()

        self.assertEqual(ai.start_calls, 0)

    async def test_status_sync_reports_camera_error_and_clears_tracks(self):
        ai = FakeAI(
            [],
            camera_status=CameraStatus(
                state=CameraState.ERROR,
                error="Unable to open configured camera",
            ),
        )
        controller = CoreController(
            ai,
            FakeSignals(),
            POLICY,
            initial_camera_status=RUNNING_CAMERA,
        )
        controller._track_guard.observe(12, 1.0)

        status = await controller.synchronize_camera_status()

        self.assertEqual(status.state, CameraState.ERROR)
        self.assertEqual(status.error, "Unable to open configured camera")
        self.assertEqual(controller._track_guard.active_track_ids(), ())

    async def test_status_transport_failure_becomes_fail_closed_camera_error(self):
        class FailingAI(FakeAI):
            async def get_camera_status(self):
                raise AIServiceClientError("unavailable")

        controller = CoreController(
            FailingAI([]),
            FakeSignals(),
            POLICY,
            initial_camera_status=RUNNING_CAMERA,
        )

        with self.assertRaises(CameraCommandError):
            await controller.synchronize_camera_status()

        self.assertEqual(controller.camera_status.state, CameraState.ERROR)

    async def test_stop_suppresses_an_in_flight_authorization_result(self):
        import asyncio

        class BlockingAI(FakeAI):
            def __init__(self):
                super().__init__(
                    [],
                    stop_result=CameraStatus(state=CameraState.STOPPING),
                )
                self.embedding_started = asyncio.Event()
                self.release_embedding = asyncio.Event()

            async def generate_embedding(self, track_id):
                self.embedding_calls += 1
                self.embedding_started.set()
                await self.release_embedding.wait()
                return EmbeddingResult(
                    track_id,
                    FaceEmbeddingStatus.OK,
                    (1.0, 0.0, 0.0),
                )

        ai = BlockingAI()
        signals = FakeSignals()
        controller = CoreController(
            ai,
            signals,
            POLICY,
            initial_camera_status=RUNNING_CAMERA,
        )
        controller.select_worksite(WORKSITE)

        authorization = asyncio.create_task(controller.handle_event(person_seen()))
        await ai.embedding_started.wait()
        await controller.stop_camera()
        ai.release_embedding.set()
        await authorization

        self.assertEqual(controller.camera_status.state, CameraState.STOPPING)
        self.assertEqual(signals.items, [])
        self.assertEqual(ai.ppe_calls, 0)


if __name__ == "__main__":
    unittest.main()
