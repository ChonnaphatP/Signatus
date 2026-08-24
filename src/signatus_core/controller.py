from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Protocol

from signatus_contracts import CameraState, CameraStatus

from .ai_client import AIServiceClientError
from .domain import (
    AIEvent,
    AIEventType,
    CoreState,
    EmbeddingResult,
    FaceEmbeddingStatus,
    GUIStatusSignal,
    OutcomeStatus,
    PPEClassRule,
    PPEResult,
    PPEResultStatus,
    Worksite,
)
from .face_match import find_best_match
from .ppe import evaluate_ppe
from .track_guard import InMemoryTrackGuard


class AICommands(Protocol):
    async def generate_embedding(self, track_id: int) -> EmbeddingResult: ...

    async def get_cached_ppe(self, track_id: int) -> PPEResult: ...

    async def get_camera_status(self) -> CameraStatus: ...

    async def start_camera(self) -> CameraStatus: ...

    async def stop_camera(self) -> CameraStatus: ...


class SignalSink(Protocol):
    async def emit(self, signal: GUIStatusSignal) -> None: ...


class CameraTransitionError(RuntimeError):
    """Raised when Core rejects an unsafe camera state transition."""


class CameraCommandError(RuntimeError):
    """Raised when Core cannot complete a validated AI camera command."""


class CoreController:
    def __init__(
        self,
        ai_commands: AICommands,
        signal_sink: SignalSink,
        ppe_policy: dict[str, PPEClassRule],
        face_match_min_cosine_similarity: float = 0.35,
        track_guard: InMemoryTrackGuard | None = None,
        clock: Callable[[], float] = time.monotonic,
        initial_camera_status: CameraStatus | None = None,
    ):
        self._ai = ai_commands
        self._signals = signal_sink
        self._ppe_policy = ppe_policy
        self._threshold = face_match_min_cosine_similarity
        self._track_guard = track_guard or InMemoryTrackGuard()
        self._clock = clock
        self._lock = asyncio.Lock()
        self._camera_command_lock = asyncio.Lock()
        self._worksite: Worksite | None = None
        self._worksite_source: str | None = None
        self._camera_status = (
            initial_camera_status.model_copy()
            if initial_camera_status is not None
            else CameraStatus(state=CameraState.STOPPED)
        )
        self._camera_epoch = 0
        self.state = CoreState.STANDBY

    @property
    def selected_worksite(self) -> Worksite | None:
        return self._worksite

    @property
    def selected_worksite_source(self) -> str | None:
        return self._worksite_source

    @property
    def camera_status(self) -> CameraStatus:
        return self._camera_status.model_copy()

    def select_worksite(self, worksite: Worksite, *, source: str | None = None) -> None:
        self._worksite = worksite
        self._worksite_source = source
        self.state = CoreState.STANDBY

    async def synchronize_camera_status(self) -> CameraStatus:
        """Refresh Core's fail-closed view of the AI-owned camera state."""

        async with self._camera_command_lock:
            try:
                status = await self._ai.get_camera_status()
            except AIServiceClientError as exc:
                status = CameraStatus(
                    state=CameraState.ERROR,
                    error="AI camera status is unavailable",
                )
                self._apply_camera_status(status)
                raise CameraCommandError(status.error) from exc
            self._apply_camera_status(status)
            return self.camera_status

    async def start_camera(self) -> CameraStatus:
        async with self._camera_command_lock:
            current = self._camera_status.state
            if current not in {CameraState.STOPPED, CameraState.ERROR}:
                raise CameraTransitionError(
                    f"cannot start camera while it is {current.value}"
                )
            self._apply_camera_status(CameraStatus(state=CameraState.STARTING))
            try:
                status = await self._ai.start_camera()
            except AIServiceClientError as exc:
                error = "AI Service did not accept the camera start command"
                self._apply_camera_status(
                    CameraStatus(state=CameraState.ERROR, error=error)
                )
                raise CameraCommandError(error) from exc
            if status.state not in {
                CameraState.STARTING,
                CameraState.RUNNING,
                CameraState.ERROR,
            }:
                error = f"AI Service returned invalid start state {status.state.value}"
                self._apply_camera_status(
                    CameraStatus(state=CameraState.ERROR, error=error)
                )
                raise CameraCommandError(error)
            self._apply_camera_status(status)
            return self.camera_status

    async def stop_camera(self) -> CameraStatus:
        async with self._camera_command_lock:
            current = self._camera_status.state
            if current not in {
                CameraState.STARTING,
                CameraState.RUNNING,
                CameraState.ERROR,
            }:
                raise CameraTransitionError(
                    f"cannot stop camera while it is {current.value}"
                )
            # Close authorization immediately, before the cross-process request.
            self._apply_camera_status(CameraStatus(state=CameraState.STOPPING))
            try:
                status = await self._ai.stop_camera()
            except AIServiceClientError as exc:
                error = "AI Service did not accept the camera stop command"
                self._apply_camera_status(
                    CameraStatus(state=CameraState.ERROR, error=error)
                )
                raise CameraCommandError(error) from exc
            if status.state not in {
                CameraState.STOPPING,
                CameraState.STOPPED,
                CameraState.ERROR,
            }:
                error = f"AI Service returned invalid stop state {status.state.value}"
                self._apply_camera_status(
                    CameraStatus(state=CameraState.ERROR, error=error)
                )
                raise CameraCommandError(error)
            self._apply_camera_status(status)
            return self.camera_status

    def report_camera_error(self, message: str) -> None:
        self._apply_camera_status(
            CameraStatus(state=CameraState.ERROR, error=message)
        )

    def _apply_camera_status(self, status: CameraStatus) -> None:
        if status == self._camera_status:
            return
        self._camera_status = status.model_copy()
        self._camera_epoch += 1
        if status.state is not CameraState.RUNNING:
            for track_id in self._track_guard.active_track_ids():
                self._track_guard.forget(track_id)

    def _camera_allows_authorization(self, epoch: int) -> bool:
        return (
            self._camera_status.state is CameraState.RUNNING
            and self._camera_epoch == epoch
        )

    async def handle_event(self, event: AIEvent) -> None:
        if event.type is AIEventType.TRACK_LOST:
            self._track_guard.forget(event.track_id)
            return

        if (
            event.type is not AIEventType.PERSON_SEEN
            or self._worksite is None
            or self._camera_status.state is not CameraState.RUNNING
        ):
            return

        async with self._lock:
            if self._camera_status.state is not CameraState.RUNNING:
                return
            now = self._clock()
            self._track_guard.observe(event.track_id, now)
            if self.state is not CoreState.STANDBY:
                return
            if not self._track_guard.should_trigger(event.track_id, now):
                return

            self.state = CoreState.AUTHORIZATION
            camera_epoch = self._camera_epoch
            try:
                await self._authorize(event.track_id, camera_epoch)
            finally:
                self.state = CoreState.STANDBY

    async def _authorize(self, track_id: int, camera_epoch: int) -> None:
        if not self._camera_allows_authorization(camera_epoch):
            return
        embedding_result = await self._ai.generate_embedding(track_id)
        if not self._camera_allows_authorization(camera_epoch):
            return
        if embedding_result.status is not FaceEmbeddingStatus.OK:
            await self._handle_face_failure(
                track_id, embedding_result.status, camera_epoch
            )
            return
        if embedding_result.embedding is None:
            await self._handle_face_failure(
                track_id, FaceEmbeddingStatus.ERROR, camera_epoch
            )
            return

        worksite = self._worksite
        if worksite is None:
            return

        worker = find_best_match(
            embedding_result.embedding,
            worksite.authorized_workers,
            self._threshold,
        )
        if worker is None:
            if not self._camera_allows_authorization(camera_epoch):
                return
            self._track_guard.mark_handled(track_id)
            await self._signals.emit(GUIStatusSignal(status=OutcomeStatus.UNAUTHORIZED))
            return

        ppe_result = await self._ai.get_cached_ppe(track_id)
        if not self._camera_allows_authorization(camera_epoch):
            return
        detected_classes = (
            ppe_result.detected_classes if ppe_result.status is PPEResultStatus.OK else ()
        )
        evaluation = evaluate_ppe(
            worksite.required_ppe,
            detected_classes,
            self._ppe_policy,
        )
        self._track_guard.mark_handled(track_id)

        if evaluation.compliant:
            signal = GUIStatusSignal(
                status=OutcomeStatus.AUTHORIZED,
                worker_id=worker.worker_id,
            )
        else:
            signal = GUIStatusSignal(
                status=OutcomeStatus.PPE_VIOLATION,
                worker_id=worker.worker_id,
                missing_ppe=evaluation.missing_ppe,
            )
        if not self._camera_allows_authorization(camera_epoch):
            return
        await self._signals.emit(signal)

    async def _handle_face_failure(
        self,
        track_id: int,
        reason: FaceEmbeddingStatus,
        camera_epoch: int,
    ) -> None:
        if not self._camera_allows_authorization(camera_epoch):
            return
        decision = self._track_guard.record_face_failure(track_id, self._clock())
        if not self._camera_allows_authorization(camera_epoch):
            return
        await self._signals.emit(
            GUIStatusSignal(
                status=OutcomeStatus.FACE_CAPTURE_FAILED,
                face_failure_reason=reason,
                attempt=decision.attempt,
                retry_allowed=decision.retry_allowed,
            )
        )
