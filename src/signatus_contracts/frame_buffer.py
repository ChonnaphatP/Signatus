"""Versioned shared-memory contract for the local camera preview.

The AI Service is the sole writer and owner of the shared-memory segment. A
display process may attach as a read-only-by-convention consumer. Screening
events and decisions do not use this transport.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from math import isfinite
from typing import Final

FRAME_BUFFER_MAGIC: Final = b"SIGNFRM\0"
FRAME_BUFFER_VERSION: Final = 2
FRAME_BUFFER_CHANNELS: Final = 3
FRAME_BUFFER_SLOT_COUNT: Final = 2
OVERLAY_SLOT_CAPACITY: Final = 65_536
MAX_DETECTIONS: Final = 1_024

# magic, version, header size, slot capacity, width, height, stride, channels,
# active slot, overlay size, seqlock sequence, capture timestamp, and padding.
HEADER_STRUCT: Final = struct.Struct("<8s9IQQ4x")
HEADER_SIZE: Final = HEADER_STRUCT.size
SEQUENCE_OFFSET: Final = 44
_DETECTION_COUNT_STRUCT: Final = struct.Struct("<I")
_DETECTION_STRUCT: Final = struct.Struct("<5fH")

_UINT32_MAX: Final = (1 << 32) - 1


class FrameBufferContractError(ValueError):
    """Raised when shared-memory metadata violates the preview contract."""


@dataclass(frozen=True, slots=True)
class FrameDetection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class FrameHeader:
    magic: bytes
    version: int
    header_size: int
    slot_capacity: int
    width: int
    height: int
    stride: int
    channels: int
    active_slot: int
    overlay_size: int
    sequence: int
    timestamp_ns: int

    @property
    def frame_size(self) -> int:
        return self.height * self.stride

    @classmethod
    def empty(cls, slot_capacity: int) -> FrameHeader:
        return cls(
            magic=FRAME_BUFFER_MAGIC,
            version=FRAME_BUFFER_VERSION,
            header_size=HEADER_SIZE,
            slot_capacity=slot_capacity,
            width=0,
            height=0,
            stride=0,
            channels=FRAME_BUFFER_CHANNELS,
            active_slot=0,
            overlay_size=0,
            sequence=0,
            timestamp_ns=0,
        )


def segment_size(slot_capacity: int) -> int:
    """Return the exact byte size of a v2 two-slot frame-and-overlay segment."""

    if not 0 < slot_capacity <= _UINT32_MAX:
        raise FrameBufferContractError("slot capacity must fit a positive uint32")
    return HEADER_SIZE + FRAME_BUFFER_SLOT_COUNT * (slot_capacity + OVERLAY_SLOT_CAPACITY)


def pack_header(header: FrameHeader) -> bytes:
    """Serialize a header without applying reader-state validation."""

    try:
        return HEADER_STRUCT.pack(
            header.magic,
            header.version,
            header.header_size,
            header.slot_capacity,
            header.width,
            header.height,
            header.stride,
            header.channels,
            header.active_slot,
            header.overlay_size,
            header.sequence,
            header.timestamp_ns,
        )
    except struct.error as exc:
        raise FrameBufferContractError(
            f"header value is outside the binary contract: {exc}"
        ) from exc


def unpack_header(buffer: object) -> FrameHeader:
    """Deserialize the fixed header at the beginning of a bytes-like object."""

    try:
        values = HEADER_STRUCT.unpack_from(buffer)  # type: ignore[arg-type]
    except (struct.error, TypeError) as exc:
        raise FrameBufferContractError("shared-memory header is unavailable or truncated") from exc
    return FrameHeader(*values)


def validate_header(
    header: FrameHeader,
    mapped_size: int,
    *,
    require_frame: bool = True,
) -> None:
    """Validate layout and stable-frame metadata, failing closed on mismatch.

    A sequence of zero denotes a newly created segment with no published frame.
    Odd sequences denote a writer in progress. Consumers must never display
    either state. A reader must additionally compare the complete header before
    and after copying a slot and accept the copy only when both are identical.
    """

    if header.magic != FRAME_BUFFER_MAGIC:
        raise FrameBufferContractError("shared-memory magic does not match")
    if header.version != FRAME_BUFFER_VERSION:
        raise FrameBufferContractError("shared-memory version is unsupported")
    if header.header_size != HEADER_SIZE:
        raise FrameBufferContractError("shared-memory header size does not match")
    expected_size = segment_size(header.slot_capacity)
    if mapped_size != expected_size:
        raise FrameBufferContractError("shared-memory segment size does not match its header")
    if header.channels != FRAME_BUFFER_CHANNELS:
        raise FrameBufferContractError("preview frame is not three-channel BGR")
    if not 0 <= header.active_slot < FRAME_BUFFER_SLOT_COUNT:
        raise FrameBufferContractError("active slot is outside the double buffer")
    if header.sequence & 1:
        raise FrameBufferContractError("preview frame is being written")

    if header.sequence == 0:
        if any(
            (
                header.width,
                header.height,
                header.stride,
                header.overlay_size,
                header.timestamp_ns,
            )
        ):
            raise FrameBufferContractError("empty frame metadata is inconsistent")
        if require_frame:
            raise FrameBufferContractError("no preview frame has been published")
        return

    if header.width <= 0 or header.height <= 0:
        raise FrameBufferContractError("preview dimensions must be positive")
    if header.stride < header.width * header.channels:
        raise FrameBufferContractError("preview stride is smaller than one pixel row")
    if header.frame_size > header.slot_capacity:
        raise FrameBufferContractError("preview frame exceeds its slot capacity")
    if not 0 <= header.overlay_size <= OVERLAY_SLOT_CAPACITY:
        raise FrameBufferContractError("preview overlay exceeds its slot capacity")
    if header.timestamp_ns <= 0:
        raise FrameBufferContractError("preview timestamp is missing")


def slot_offset(header: FrameHeader, slot: int | None = None) -> int:
    """Return the byte offset for an explicit slot or the active slot."""

    selected = header.active_slot if slot is None else slot
    if not 0 <= selected < FRAME_BUFFER_SLOT_COUNT:
        raise FrameBufferContractError("slot is outside the double buffer")
    return HEADER_SIZE + selected * header.slot_capacity


def overlay_offset(header: FrameHeader, slot: int | None = None) -> int:
    selected = header.active_slot if slot is None else slot
    if not 0 <= selected < FRAME_BUFFER_SLOT_COUNT:
        raise FrameBufferContractError("slot is outside the double buffer")
    return HEADER_SIZE + FRAME_BUFFER_SLOT_COUNT * header.slot_capacity + (
        selected * OVERLAY_SLOT_CAPACITY
    )


def encode_detections(detections: tuple[FrameDetection, ...]) -> bytes:
    if len(detections) > MAX_DETECTIONS:
        raise FrameBufferContractError("preview contains too many detections")
    payload = bytearray(_DETECTION_COUNT_STRUCT.pack(len(detections)))
    for detection in detections:
        name = detection.class_name.strip().encode("utf-8")
        values = (
            detection.confidence,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        )
        if not name or len(name) > 255:
            raise FrameBufferContractError("preview detection class name is invalid")
        if not all(isfinite(value) for value in values):
            raise FrameBufferContractError("preview detection contains a non-finite value")
        if not 0.0 <= detection.confidence <= 1.0:
            raise FrameBufferContractError("preview detection confidence is invalid")
        payload.extend(
            _DETECTION_STRUCT.pack(
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                detection.confidence,
                len(name),
            )
        )
        payload.extend(name)
    if len(payload) > OVERLAY_SLOT_CAPACITY:
        raise FrameBufferContractError("preview overlay exceeds its slot capacity")
    return bytes(payload)


def decode_detections(payload: bytes) -> tuple[FrameDetection, ...]:
    if len(payload) < _DETECTION_COUNT_STRUCT.size:
        raise FrameBufferContractError("preview overlay is truncated")
    (count,) = _DETECTION_COUNT_STRUCT.unpack_from(payload)
    if count > MAX_DETECTIONS:
        raise FrameBufferContractError("preview contains too many detections")
    offset = _DETECTION_COUNT_STRUCT.size
    detections: list[FrameDetection] = []
    for _index in range(count):
        if offset + _DETECTION_STRUCT.size > len(payload):
            raise FrameBufferContractError("preview detection is truncated")
        x1, y1, x2, y2, confidence, name_size = _DETECTION_STRUCT.unpack_from(payload, offset)
        offset += _DETECTION_STRUCT.size
        end = offset + name_size
        if name_size == 0 or end > len(payload):
            raise FrameBufferContractError("preview detection class name is truncated")
        try:
            class_name = payload[offset:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FrameBufferContractError("preview detection class name is invalid") from exc
        detection = FrameDetection(class_name, confidence, x1, y1, x2, y2)
        # Apply the writer-side semantic validation to untrusted shared memory.
        encode_detections((detection,))
        detections.append(detection)
        offset = end
    if offset != len(payload):
        raise FrameBufferContractError("preview overlay has trailing data")
    return tuple(detections)
