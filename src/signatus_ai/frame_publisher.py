"""AI-owned shared-memory publisher for the local display preview."""

from __future__ import annotations

import logging
import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Self

from signatus_contracts.frame_buffer import (
    HEADER_SIZE,
    SEQUENCE_OFFSET,
    FrameDetection,
    FrameHeader,
    encode_detections,
    overlay_offset,
    pack_header,
    segment_size,
    slot_offset,
)

logger = logging.getLogger(__name__)

_SEQUENCE_STRUCT = struct.Struct("<Q")
_MAX_STABLE_SEQUENCE = (1 << 64) - 2


class SharedMemoryFramePublisher:
    """Publish latest BGR frames through an AI-owned double buffer.

    Preview failures are deliberately non-fatal: callers receive ``False`` and
    tracking can continue. This object is the sole writer and unlinks only a
    segment it successfully created itself.
    """

    def __init__(self, name: str, slot_capacity: int, *, enabled: bool = True):
        cleaned_name = name.strip()
        if enabled and not cleaned_name:
            raise ValueError("shared-memory name must not be empty when preview is enabled")
        segment_size(slot_capacity)

        self._name = cleaned_name
        self._slot_capacity = slot_capacity
        self._enabled = enabled
        self._shared_memory: SharedMemory | None = None
        self._active_slot = 0
        self._sequence = 0
        self._last_publish_timestamp_ns = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def slot_capacity(self) -> int:
        return self._slot_capacity

    @property
    def is_open(self) -> bool:
        return self._shared_memory is not None

    @property
    def last_publish_timestamp_ns(self) -> int:
        return self._last_publish_timestamp_ns

    def open(self) -> bool:
        """Create and initialize the segment, returning whether it is ready."""

        with self._lock:
            if not self._enabled:
                return False
            if self._shared_memory is not None:
                return True

            try:
                shared_memory = SharedMemory(
                    name=self._name,
                    create=True,
                    size=segment_size(self._slot_capacity),
                )
            except (FileExistsError, OSError, ValueError) as exc:
                logger.warning(
                    "Camera preview shared memory %r could not be created; "
                    "tracking will continue without preview: %s",
                    self._name,
                    exc,
                )
                return False

            try:
                shared_memory.buf[:HEADER_SIZE] = pack_header(
                    FrameHeader.empty(self._slot_capacity)
                )
            except Exception:
                logger.exception(
                    "Camera preview shared memory %r could not be initialized; "
                    "tracking will continue without preview",
                    self._name,
                )
                try:
                    shared_memory.unlink()
                except FileNotFoundError:
                    pass
                shared_memory.close()
                return False

            self._shared_memory = shared_memory
            self._active_slot = 0
            self._sequence = 0
            self._last_publish_timestamp_ns = 0
            return True

    def publish(
        self,
        frame: Any,
        *,
        timestamp_ns: int | None = None,
        detections: tuple[FrameDetection, ...] = (),
    ) -> bool:
        """Publish one uint8 HxWx3 BGR frame, returning success."""

        with self._lock:
            shared_memory = self._shared_memory
            if shared_memory is None:
                return False

            try:
                import numpy as np

                array = np.asarray(frame)
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                    logger.error("Camera preview rejected a frame that is not uint8 HxWx3 BGR")
                    return False
                if not array.flags.c_contiguous:
                    array = np.ascontiguousarray(array)

                height, width, channels = (int(value) for value in array.shape)
                stride = int(array.strides[0])
                frame_size = height * stride
                if width <= 0 or height <= 0:
                    logger.error("Camera preview rejected an empty frame")
                    return False
                if frame_size > self._slot_capacity:
                    logger.error(
                        "Camera preview frame requires %d bytes but the slot capacity is %d",
                        frame_size,
                        self._slot_capacity,
                    )
                    return False

                captured_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
                if captured_ns <= 0 or captured_ns >= 1 << 64:
                    logger.error("Camera preview rejected an invalid capture timestamp")
                    return False
                overlay = encode_detections(detections)

                odd_sequence, even_sequence = self._next_sequences()
                inactive_slot = 1 - self._active_slot
                buffer = shared_memory.buf
                empty_header = FrameHeader.empty(self._slot_capacity)

                # Mark the segment inconsistent before touching the inactive
                # slot. A consumer accepts only equal, even headers read before
                # and after its slot copy.
                _SEQUENCE_STRUCT.pack_into(buffer, SEQUENCE_OFFSET, odd_sequence)

                payload = memoryview(array).cast("B")
                try:
                    start = slot_offset(empty_header, inactive_slot)
                    buffer[start : start + frame_size] = payload
                finally:
                    payload.release()
                metadata_start = overlay_offset(empty_header, inactive_slot)
                buffer[metadata_start : metadata_start + len(overlay)] = overlay

                pending_header = FrameHeader(
                    magic=empty_header.magic,
                    version=empty_header.version,
                    header_size=HEADER_SIZE,
                    slot_capacity=self._slot_capacity,
                    width=width,
                    height=height,
                    stride=stride,
                    channels=channels,
                    active_slot=inactive_slot,
                    overlay_size=len(overlay),
                    sequence=odd_sequence,
                    timestamp_ns=captured_ns,
                )
                buffer[:HEADER_SIZE] = pack_header(pending_header)
                _SEQUENCE_STRUCT.pack_into(buffer, SEQUENCE_OFFSET, even_sequence)
            except Exception:
                # Preview is observational only. A malformed frame or a closed
                # mapping must never stop tracking or change screening state.
                logger.exception("Camera preview frame publish failed")
                return False

            self._active_slot = inactive_slot
            self._sequence = even_sequence
            self._last_publish_timestamp_ns = captured_ns
            return True

    def close(self) -> None:
        """Unlink and close the segment if this publisher created it."""

        with self._lock:
            shared_memory = self._shared_memory
            self._shared_memory = None
            self._active_slot = 0
            self._sequence = 0
            self._last_publish_timestamp_ns = 0
            if shared_memory is None:
                return

            try:
                shared_memory.unlink()
            except FileNotFoundError:
                pass
            finally:
                shared_memory.close()

    def invalidate(self) -> None:
        """Make the last preview unreadable while retaining segment ownership."""

        with self._lock:
            shared_memory = self._shared_memory
            self._active_slot = 0
            self._sequence = 0
            self._last_publish_timestamp_ns = 0
            if shared_memory is None:
                return
            # An odd sequence makes concurrent readers reject the old slot;
            # the empty header then exposes no dimensions, overlay, or frame.
            _SEQUENCE_STRUCT.pack_into(shared_memory.buf, SEQUENCE_OFFSET, 1)
            shared_memory.buf[:HEADER_SIZE] = pack_header(
                FrameHeader.empty(self._slot_capacity)
            )

    def _next_sequences(self) -> tuple[int, int]:
        if self._sequence >= _MAX_STABLE_SEQUENCE:
            return 1, 2
        return self._sequence + 1, self._sequence + 2

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
