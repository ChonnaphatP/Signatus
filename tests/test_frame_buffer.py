from __future__ import annotations

import unittest
from dataclasses import replace

from signatus_contracts.frame_buffer import (
    FRAME_BUFFER_CHANNELS,
    FRAME_BUFFER_MAGIC,
    FRAME_BUFFER_VERSION,
    HEADER_SIZE,
    HEADER_STRUCT,
    FrameBufferContractError,
    FrameDetection,
    FrameHeader,
    decode_detections,
    encode_detections,
    overlay_offset,
    pack_header,
    segment_size,
    slot_offset,
    unpack_header,
    validate_header,
)

CAPACITY = 4096


def valid_header() -> FrameHeader:
    return FrameHeader(
        magic=FRAME_BUFFER_MAGIC,
        version=FRAME_BUFFER_VERSION,
        header_size=HEADER_SIZE,
        slot_capacity=CAPACITY,
        width=20,
        height=10,
        stride=60,
        channels=FRAME_BUFFER_CHANNELS,
        active_slot=1,
        overlay_size=4,
        sequence=2,
        timestamp_ns=1_750_000_000_000_000_000,
    )


class FrameBufferContractTests(unittest.TestCase):
    def test_v2_header_is_exactly_64_bytes_and_round_trips(self) -> None:
        header = valid_header()

        self.assertEqual(HEADER_STRUCT.format, "<8s9IQQ4x")
        self.assertEqual(HEADER_SIZE, 64)
        self.assertEqual(unpack_header(pack_header(header)), header)

    def test_valid_header_and_slot_offsets(self) -> None:
        header = valid_header()

        validate_header(header, segment_size(CAPACITY))

        self.assertEqual(slot_offset(header, 0), HEADER_SIZE)
        self.assertEqual(slot_offset(header), HEADER_SIZE + CAPACITY)
        self.assertEqual(overlay_offset(header, 0), HEADER_SIZE + 2 * CAPACITY)

    def test_detection_metadata_round_trips(self) -> None:
        detections = (
            FrameDetection("helmet", 0.94, 10.0, 20.0, 80.0, 90.0),
            FrameDetection("person", 0.97, 1.0, 2.0, 100.0, 200.0),
        )

        decoded = decode_detections(encode_detections(detections))

        self.assertEqual([item.class_name for item in decoded], ["helmet", "person"])
        self.assertAlmostEqual(decoded[0].confidence, 0.94, places=5)

    def test_invalid_detection_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(FrameBufferContractError, "confidence"):
            encode_detections((FrameDetection("helmet", 1.2, 0, 0, 1, 1),))
        with self.assertRaisesRegex(FrameBufferContractError, "truncated"):
            decode_detections(b"\x01\x00\x00\x00")

    def test_empty_header_is_valid_layout_but_not_a_displayable_frame(self) -> None:
        header = FrameHeader.empty(CAPACITY)

        validate_header(header, segment_size(CAPACITY), require_frame=False)
        with self.assertRaisesRegex(FrameBufferContractError, "no preview frame"):
            validate_header(header, segment_size(CAPACITY))

    def test_odd_sequence_is_rejected_as_an_in_progress_write(self) -> None:
        with self.assertRaisesRegex(FrameBufferContractError, "being written"):
            validate_header(replace(valid_header(), sequence=3), segment_size(CAPACITY))

    def test_corrupt_layout_metadata_is_rejected(self) -> None:
        cases = (
            (replace(valid_header(), magic=b"CORRUPT\0"), "magic"),
            (replace(valid_header(), version=1), "version"),
            (replace(valid_header(), header_size=32), "header size"),
            (replace(valid_header(), channels=4), "three-channel"),
            (replace(valid_header(), active_slot=2), "active slot"),
            (replace(valid_header(), stride=59), "stride"),
            (replace(valid_header(), height=100), "slot capacity"),
            (replace(valid_header(), timestamp_ns=0), "timestamp"),
        )

        for header, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(FrameBufferContractError, message),
            ):
                validate_header(header, segment_size(CAPACITY))

    def test_segment_size_mismatch_and_truncated_header_are_rejected(self) -> None:
        with self.assertRaisesRegex(FrameBufferContractError, "segment size"):
            validate_header(valid_header(), segment_size(CAPACITY) - 1)
        with self.assertRaisesRegex(FrameBufferContractError, "truncated"):
            unpack_header(b"too short")

    def test_invalid_capacity_and_slot_are_rejected(self) -> None:
        for capacity in (0, -1, 1 << 32):
            with self.subTest(capacity=capacity), self.assertRaises(FrameBufferContractError):
                segment_size(capacity)

        with self.assertRaisesRegex(FrameBufferContractError, "slot"):
            slot_offset(valid_header(), 2)


if __name__ == "__main__":
    unittest.main()
