from __future__ import annotations

import importlib.util
import unittest
import uuid
from unittest.mock import patch

from signatus_contracts.frame_buffer import (
    FRAME_BUFFER_CHANNELS,
    FRAME_BUFFER_MAGIC,
    FRAME_BUFFER_VERSION,
    HEADER_SIZE,
    FrameHeader,
    encode_detections,
    overlay_offset,
    pack_header,
    segment_size,
    slot_offset,
)

PYSIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None
if PYSIDE_AVAILABLE:
    from signatus_gui.frame_reader import (
        CameraFrame,
        SharedMemoryFrameReader,
        _copy_stable_frame,
        _to_detached_qimage,
    )


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 GUI extra is not installed")
class SharedMemoryFrameReaderTests(unittest.TestCase):
    def test_copies_only_a_stable_active_slot(self) -> None:
        capacity = 24
        memory = bytearray(segment_size(capacity))
        header = _header(capacity=capacity, active_slot=1, sequence=2)
        expected = bytes(range(header.frame_size))
        start = slot_offset(header)
        memory[start : start + header.frame_size] = expected
        overlay = encode_detections(())
        metadata_start = overlay_offset(header)
        memory[metadata_start : metadata_start + len(overlay)] = overlay
        memory[:HEADER_SIZE] = pack_header(header)

        frame = _copy_stable_frame(memoryview(memory), len(memory))

        self.assertIsNotNone(frame)
        self.assertEqual(frame.sequence, 2)
        self.assertEqual(frame.pixels, expected)

    def test_never_reads_an_in_progress_frame(self) -> None:
        capacity = 24
        memory = bytearray(segment_size(capacity))
        memory[:HEADER_SIZE] = pack_header(_header(capacity=capacity, active_slot=1, sequence=3))

        self.assertIsNone(_copy_stable_frame(memoryview(memory), len(memory)))

    def test_missing_segment_is_an_unavailable_frame_not_an_exception(self) -> None:
        reader = SharedMemoryFrameReader(
            f"signatus-missing-{uuid.uuid4().hex}", attach_retry_seconds=0
        )
        self.addCleanup(reader.close)

        self.assertIsNone(reader.read_latest())
        self.assertFalse(reader.attached)

    def test_repeated_reads_keep_shared_memory_mapping_usable(self) -> None:
        capacity = 24
        memory = bytearray(segment_size(capacity))
        segment = _PersistentBufferSegment(memory)
        reader = SharedMemoryFrameReader("signatus-reader-test", attach_retry_seconds=0)
        header = _header(capacity=capacity, active_slot=1, sequence=2)
        expected = bytes(range(header.frame_size))
        start = slot_offset(header)
        memory[start : start + header.frame_size] = expected
        overlay = encode_detections(())
        metadata_start = overlay_offset(header)
        memory[metadata_start : metadata_start + len(overlay)] = overlay
        memory[:HEADER_SIZE] = pack_header(header)

        try:
            with patch("signatus_gui.frame_reader._attach_non_owner", return_value=segment):
                first = reader.read_latest()
                second = reader.read_latest()

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertTrue(reader.attached)
            self.assertEqual(first.pixels, expected)
            self.assertEqual(second.pixels, expected)
        finally:
            reader.close()

        self.assertTrue(segment.closed)

    def test_qimage_owns_its_copy_of_frame_bytes(self) -> None:
        source = bytearray((0, 0, 255, 0, 255, 0))
        frame = CameraFrame(
            width=2,
            height=1,
            stride=6,
            sequence=2,
            timestamp_ns=1,
            pixels=source,
        )

        image = _to_detached_qimage(frame)
        source[:] = b"\0" * len(source)

        self.assertEqual(image.pixelColor(0, 0).getRgb()[:3], (255, 0, 0))
        self.assertEqual(image.pixelColor(1, 0).getRgb()[:3], (0, 255, 0))


def _header(*, capacity: int, active_slot: int, sequence: int) -> FrameHeader:
    return FrameHeader(
        magic=FRAME_BUFFER_MAGIC,
        version=FRAME_BUFFER_VERSION,
        header_size=HEADER_SIZE,
        slot_capacity=capacity,
        width=2,
        height=2,
        stride=8,
        channels=FRAME_BUFFER_CHANNELS,
        active_slot=active_slot,
        overlay_size=len(encode_detections(())),
        sequence=sequence,
        timestamp_ns=1,
    )


class _PersistentBufferSegment:
    """Model SharedMemory.buf returning the same owned view on every access."""

    def __init__(self, memory: bytearray) -> None:
        self.buf = memoryview(memory)
        self.size = len(memory)
        self.closed = False

    def close(self) -> None:
        self.buf.release()
        self.closed = True


if __name__ == "__main__":
    unittest.main()
