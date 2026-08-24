"""Read the AI Service's latest preview frame without owning camera state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

from signatus_contracts.frame_buffer import (
    FrameBufferContractError,
    FrameDetection,
    decode_detections,
    overlay_offset,
    slot_offset,
    unpack_header,
    validate_header,
)


@dataclass(frozen=True, slots=True)
class CameraFrame:
    width: int
    height: int
    stride: int
    sequence: int
    timestamp_ns: int
    pixels: bytes
    detections: tuple[FrameDetection, ...] = ()


class SharedMemoryFrameReader:
    """Non-owning, read-only-by-convention view of the preview segment."""

    def __init__(self, name: str, *, attach_retry_seconds: float = 1.0) -> None:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("shared-memory name must not be empty")
        if attach_retry_seconds < 0:
            raise ValueError("shared-memory attach retry must not be negative")
        self._name = cleaned_name
        self._attach_retry_seconds = attach_retry_seconds
        self._segment: shared_memory.SharedMemory | None = None
        self._next_attach_at = 0.0

    @property
    def attached(self) -> bool:
        return self._segment is not None

    def read_latest(self) -> CameraFrame | None:
        if not self._ensure_attached():
            return None
        segment = self._segment
        if segment is None:
            return None
        try:
            # SharedMemory.buf is the segment's persistent memoryview, not a
            # disposable view created for each access. Releasing it here makes
            # every subsequent poll fail with a released-memoryview error and
            # forces the preview into its unavailable/reconnect cycle. The
            # SharedMemory object releases the view when disconnect() closes it.
            return _copy_stable_frame(segment.buf, segment.size)
        except (BufferError, FrameBufferContractError, OSError, ValueError):
            self.disconnect()
            self._next_attach_at = time.monotonic() + self._attach_retry_seconds
            return None

    def disconnect(self) -> None:
        segment, self._segment = self._segment, None
        if segment is not None:
            try:
                segment.close()
            except BufferError:
                # No shared-memory views are retained, but shutdown must remain
                # harmless even if an interpreter-owned view outlives this call.
                pass
        self._next_attach_at = 0.0

    close = disconnect

    def _ensure_attached(self) -> bool:
        if self._segment is not None:
            return True
        now = time.monotonic()
        if now < self._next_attach_at:
            return False
        try:
            self._segment = _attach_non_owner(self._name)
        except (FileNotFoundError, OSError):
            self._next_attach_at = now + self._attach_retry_seconds
            return False
        return True


class SharedMemoryPreview(QObject):
    """Poll shared memory on the Qt thread and emit detached QImages."""

    frame_ready = Signal(object)
    availability_changed = Signal(bool)

    def __init__(
        self,
        shared_memory_name: str,
        parent: QObject | None = None,
        *,
        poll_interval_ms: int = 33,
        stale_after_seconds: float = 3.0,
    ) -> None:
        super().__init__(parent)
        if poll_interval_ms <= 0:
            raise ValueError("preview poll interval must be positive")
        if stale_after_seconds <= 0:
            raise ValueError("preview stale timeout must be positive")
        self._reader = SharedMemoryFrameReader(shared_memory_name)
        self._stale_after_seconds = stale_after_seconds
        self._last_sequence: int | None = None
        self._last_frame_at: float | None = None
        self._waiting_since: float | None = None
        self._available: bool | None = None
        self._started = False
        self._camera_running = False
        self._timer = QTimer(self)
        self._timer.setInterval(poll_interval_ms)
        self._timer.timeout.connect(self._poll)

    def start(self) -> None:
        self._started = True
        self._set_available(False)
        if self._camera_running:
            self._poll()
            self._timer.start()

    def set_camera_running(self, running: bool) -> None:
        if running == self._camera_running:
            return
        self._camera_running = running
        if running and self._started:
            self._poll()
            self._timer.start()
            return
        self._timer.stop()
        self._reader.disconnect()
        self._forget_frame()
        self._set_available(False)

    def close(self) -> None:
        self._started = False
        self._timer.stop()
        self._reader.close()

    def _poll(self) -> None:
        if not self._camera_running:
            return
        now = time.monotonic()
        frame = self._reader.read_latest()
        if frame is not None and frame.sequence != self._last_sequence:
            image = _to_detached_qimage(frame)
            if not image.isNull():
                self._last_sequence = frame.sequence
                self._last_frame_at = now
                self._waiting_since = None
                self.frame_ready.emit((image, frame.detections))
                self._set_available(True)
                return

        if not self._reader.attached:
            self._forget_frame()
            self._set_available(False)
            return

        if self._last_frame_at is None:
            if self._waiting_since is None:
                self._waiting_since = now
            elif now - self._waiting_since >= self._stale_after_seconds:
                self._reader.disconnect()
                self._forget_frame()
            self._set_available(False)
        elif now - self._last_frame_at >= self._stale_after_seconds:
            # An unlinked segment remains mapped in existing readers. Closing a
            # stale mapping lets the GUI attach to a replacement from a restarted
            # AI Service without ever owning or unlinking the segment itself.
            self._reader.disconnect()
            self._forget_frame()
            self._set_available(False)

    def _forget_frame(self) -> None:
        self._last_sequence = None
        self._last_frame_at = None
        self._waiting_since = None

    def _set_available(self, available: bool) -> None:
        if available == self._available:
            return
        self._available = available
        self.availability_changed.emit(available)


def _copy_stable_frame(buffer: memoryview, mapped_size: int) -> CameraFrame | None:
    for _attempt in range(3):
        before = unpack_header(buffer)
        if before.sequence == 0 or before.sequence & 1:
            return None
        validate_header(before, mapped_size)

        start = slot_offset(before)
        pixels = bytes(buffer[start : start + before.frame_size])
        metadata_start = overlay_offset(before)
        overlay = bytes(buffer[metadata_start : metadata_start + before.overlay_size])
        detections = decode_detections(overlay)

        after = unpack_header(buffer)
        if before == after and not after.sequence & 1:
            return CameraFrame(
                width=before.width,
                height=before.height,
                stride=before.stride,
                sequence=before.sequence,
                timestamp_ns=before.timestamp_ns,
                pixels=pixels,
                detections=detections,
            )
    return None


def _to_detached_qimage(frame: CameraFrame) -> QImage:
    image = QImage(
        frame.pixels,
        frame.width,
        frame.height,
        frame.stride,
        QImage.Format.Format_BGR888,
    )
    return image.copy()


def _attach_non_owner(name: str) -> shared_memory.SharedMemory:
    """Attach without allowing the consumer's resource tracker to unlink it."""

    try:
        return shared_memory.SharedMemory(name=name, create=False, track=False)
    except TypeError:
        # Python 3.11/3.12 do not expose ``track=False``. Unregistering only the
        # consumer-side tracker preserves the AI Service's ownership and cleanup.
        segment = shared_memory.SharedMemory(name=name, create=False)
        resource_tracker.unregister(segment._name, "shared_memory")
        return segment
