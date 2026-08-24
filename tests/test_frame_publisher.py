from __future__ import annotations

import unittest
import uuid
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from signatus_ai.frame_publisher import SharedMemoryFramePublisher
from signatus_contracts.frame_buffer import (
    FrameDetection,
    FrameHeader,
    decode_detections,
    overlay_offset,
    segment_size,
    slot_offset,
    unpack_header,
    validate_header,
)


def unique_name() -> str:
    return f"signatus_test_{uuid.uuid4().hex}"


class SharedMemoryFramePublisherTests(unittest.TestCase):
    def test_publishes_stable_frames_to_alternating_slots(self) -> None:
        name = unique_name()
        publisher = SharedMemoryFramePublisher(name, 256)
        reader: SharedMemory | None = None
        try:
            self.assertTrue(publisher.open())
            reader = SharedMemory(name=name, create=False)

            empty = unpack_header(reader.buf)
            validate_header(empty, reader.size, require_frame=False)
            self.assertEqual(empty.sequence, 0)

            first = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            detection = FrameDetection("helmet", 0.94, 1, 2, 3, 4)
            self.assertTrue(
                publisher.publish(first, timestamp_ns=101, detections=(detection,))
            )
            first_header = self._stable_header(reader)
            self.assertEqual(first_header.active_slot, 1)
            self.assertEqual(first_header.sequence, 2)
            self.assertEqual(first_header.timestamp_ns, 101)
            self.assertEqual(publisher.last_publish_timestamp_ns, 101)
            self.assertEqual((first_header.width, first_header.height), (4, 3))
            self.assertEqual(first_header.stride, 12)
            self.assertEqual(self._frame_bytes(reader, first_header), first.tobytes())
            overlay_start = overlay_offset(first_header)
            overlays = decode_detections(
                bytes(reader.buf[overlay_start : overlay_start + first_header.overlay_size])
            )
            self.assertEqual(overlays[0].class_name, "helmet")
            self.assertAlmostEqual(overlays[0].confidence, 0.94, places=5)

            second = np.full((2, 5, 3), 71, dtype=np.uint8)
            self.assertTrue(publisher.publish(second, timestamp_ns=202))
            second_header = self._stable_header(reader)
            self.assertEqual(second_header.active_slot, 0)
            self.assertEqual(second_header.sequence, 4)
            self.assertEqual(second_header.timestamp_ns, 202)
            self.assertEqual(publisher.last_publish_timestamp_ns, 202)
            self.assertEqual(self._frame_bytes(reader, second_header), second.tobytes())
        finally:
            if reader is not None:
                reader.close()
            publisher.close()

        self.assertEqual(publisher.last_publish_timestamp_ns, 0)

        with self.assertRaises(FileNotFoundError):
            SharedMemory(name=name, create=False)

    def test_non_contiguous_frame_is_copied_as_contiguous_bgr(self) -> None:
        name = unique_name()
        publisher = SharedMemoryFramePublisher(name, 256)
        reader: SharedMemory | None = None
        try:
            self.assertTrue(publisher.open())
            reader = SharedMemory(name=name, create=False)
            source = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)[:, ::2, :]
            self.assertFalse(source.flags.c_contiguous)

            self.assertTrue(publisher.publish(source, timestamp_ns=303))

            header = self._stable_header(reader)
            self.assertEqual(header.stride, header.width * 3)
            self.assertEqual(
                self._frame_bytes(reader, header), np.ascontiguousarray(source).tobytes()
            )
        finally:
            if reader is not None:
                reader.close()
            publisher.close()

    def test_oversized_or_invalid_frame_does_not_replace_last_good_frame(self) -> None:
        name = unique_name()
        publisher = SharedMemoryFramePublisher(name, 24)
        reader: SharedMemory | None = None
        try:
            self.assertTrue(publisher.open())
            reader = SharedMemory(name=name, create=False)
            good = np.zeros((2, 4, 3), dtype=np.uint8)
            self.assertTrue(publisher.publish(good, timestamp_ns=404))
            before = unpack_header(reader.buf)

            self.assertFalse(
                publisher.publish(np.zeros((3, 4, 3), dtype=np.uint8), timestamp_ns=505)
            )
            self.assertFalse(
                publisher.publish(np.zeros((2, 4, 4), dtype=np.uint8), timestamp_ns=606)
            )

            self.assertEqual(unpack_header(reader.buf), before)
        finally:
            if reader is not None:
                reader.close()
            publisher.close()

    def test_disabled_or_name_conflicted_publisher_fails_without_taking_ownership(self) -> None:
        disabled = SharedMemoryFramePublisher("", 32, enabled=False)
        self.assertFalse(disabled.open())
        self.assertFalse(disabled.publish(np.zeros((1, 1, 3), dtype=np.uint8)))
        disabled.close()

        name = unique_name()
        existing = SharedMemory(name=name, create=True, size=segment_size(32))
        publisher = SharedMemoryFramePublisher(name, 32)
        try:
            self.assertFalse(publisher.open())
            self.assertFalse(publisher.is_open)
            publisher.close()

            probe = SharedMemory(name=name, create=False)
            probe.close()
        finally:
            existing.unlink()
            existing.close()

    def test_invalidate_clears_stale_frame_but_keeps_owned_segment_for_restart(self) -> None:
        name = unique_name()
        publisher = SharedMemoryFramePublisher(name, 64)
        reader: SharedMemory | None = None
        try:
            self.assertTrue(publisher.open())
            reader = SharedMemory(name=name, create=False)
            self.assertTrue(
                publisher.publish(
                    np.ones((2, 3, 3), dtype=np.uint8),
                    timestamp_ns=707,
                )
            )

            publisher.invalidate()

            empty = unpack_header(reader.buf)
            validate_header(empty, reader.size, require_frame=False)
            self.assertEqual(empty.sequence, 0)
            self.assertEqual(empty.timestamp_ns, 0)
            self.assertEqual(empty.frame_size, 0)
            self.assertTrue(publisher.is_open)
            self.assertEqual(publisher.last_publish_timestamp_ns, 0)
            self.assertTrue(
                publisher.publish(
                    np.full((1, 2, 3), 9, dtype=np.uint8),
                    timestamp_ns=808,
                )
            )
            self.assertEqual(self._stable_header(reader).timestamp_ns, 808)
        finally:
            if reader is not None:
                reader.close()
            publisher.close()

        with self.assertRaises(FileNotFoundError):
            SharedMemory(name=name, create=False)

    @staticmethod
    def _stable_header(reader: SharedMemory) -> FrameHeader:
        first = unpack_header(reader.buf)
        validate_header(first, reader.size)
        second = unpack_header(reader.buf)
        if first != second:
            raise AssertionError("header changed while test reader inspected it")
        return first

    @staticmethod
    def _frame_bytes(reader: SharedMemory, header: FrameHeader) -> bytes:
        start = slot_offset(header)
        return bytes(reader.buf[start : start + header.frame_size])


if __name__ == "__main__":
    unittest.main()
