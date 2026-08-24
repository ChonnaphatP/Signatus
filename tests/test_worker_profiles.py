from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from signatus_core.worker_profiles import (
    WorkerProfileRepository,
    decode_face_image_data_uri,
    encode_face_image,
    load_worker_profile_file,
    parse_worker_profile_json,
    sanitize_profile_filename,
    validate_worker_profile,
)

JPEG_BYTES = b"\xff\xd8\xff\xe0synthetic-jpeg-data"
PNG_BYTES = b"\x89PNG\r\n\x1a\nsynthetic-png-data"


def _profile(
    worker_id: str = "W001",
    *,
    name: str = "Example Worker",
    face_image: object = None,
) -> dict[str, object]:
    return {
        "worker_id": worker_id,
        "name": name,
        "face_image": (
            encode_face_image(JPEG_BYTES) if face_image is None else face_image
        ),
    }


class WorkerProfileValidationTests(unittest.TestCase):
    def test_valid_profile_is_normalized_to_plain_contract(self) -> None:
        result = validate_worker_profile(_profile(), source="worker.json")

        self.assertTrue(result.valid)
        assert result.profile is not None
        self.assertEqual(result.profile.worker_id, "W001")
        self.assertEqual(
            set(result.profile.to_dict()), {"worker_id", "name", "face_image"}
        )
        self.assertEqual(result.to_dict()["profile"], result.profile.to_dict())

    def test_requires_nonempty_identity_name_and_only_v1_fields(self) -> None:
        payload = _profile()
        payload.update(worker_id=" ", name="", department="Operations")

        result = validate_worker_profile(payload)

        self.assertFalse(result.valid)
        self.assertEqual(
            {error.code for error in result.errors},
            {
                "INVALID_WORKER_ID",
                "INVALID_WORKER_NAME",
                "UNEXPECTED_PROFILE_FIELDS",
            },
        )
        self.assertTrue(all(isinstance(error.to_dict(), dict) for error in result.errors))

    def test_embedding_is_not_part_of_worker_profile_schema(self) -> None:
        payload = _profile()
        payload["embedding"] = [1.0] + [0.0] * 127

        result = validate_worker_profile(payload)

        self.assertFalse(result.valid)
        self.assertEqual(
            {error.code for error in result.errors}, {"UNEXPECTED_PROFILE_FIELDS"}
        )

    def test_face_image_requires_complete_supported_data_uri(self) -> None:
        cases = (
            "not-a-data-uri",
            "data:image/jpeg;base64,",
            "data:text/plain;base64,SGVsbG8=",
            "data:image/png;base64," + encode_face_image(JPEG_BYTES).partition(",")[2],
        )

        for face_image in cases:
            with self.subTest(face_image=face_image[:30]):
                result = validate_worker_profile(_profile(face_image=face_image))
                self.assertFalse(result.valid)
                self.assertIn("INVALID_FACE_IMAGE", {error.code for error in result.errors})

    def test_face_image_size_limit_matches_ai_enrollment_contract(self) -> None:
        oversized = encode_face_image(PNG_BYTES + b"x" * 100)

        with (
            patch("signatus_core.worker_profiles.MAX_ENROLLMENT_IMAGE_BYTES", 8),
            patch("signatus_core.worker_profiles._MAX_ENCODED_IMAGE_CHARACTERS", 12),
        ):
            result = validate_worker_profile(_profile(face_image=oversized))

        self.assertFalse(result.valid)
        self.assertIn("INVALID_FACE_IMAGE", {error.code for error in result.errors})
        self.assertIn("must not exceed 8", result.errors[0].message)

    def test_encode_detects_jpeg_and_png_and_rejects_mime_mismatch(self) -> None:
        jpeg_uri = encode_face_image(JPEG_BYTES, mime_type="image/jpeg")
        png_uri = encode_face_image(PNG_BYTES)

        self.assertTrue(jpeg_uri.startswith("data:image/jpeg;base64,"))
        self.assertTrue(png_uri.startswith("data:image/png;base64,"))
        self.assertEqual(decode_face_image_data_uri(jpeg_uri), ("image/jpeg", JPEG_BYTES))
        with self.assertRaisesRegex(ValueError, "does not match"):
            encode_face_image(JPEG_BYTES, mime_type="image/png")
        with self.assertRaisesRegex(ValueError, "supported"):
            encode_face_image(b"unknown image bytes")

    def test_json_and_external_file_loaders_report_parse_errors(self) -> None:
        parsed = parse_worker_profile_json("{broken", source="broken.json")
        self.assertFalse(parsed.valid)
        self.assertEqual(parsed.errors[0].code, "INVALID_WORKER_PROFILE_JSON")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(_profile()), encoding="utf-8")
            loaded = load_worker_profile_file(path)

        self.assertTrue(loaded.valid)


class WorkerProfileRepositoryTests(unittest.TestCase):
    def test_create_uses_safe_filename_and_atomic_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = WorkerProfileRepository(root)

            result = repository.create(_profile("PMII / W001"))

            self.assertTrue(result.success)
            self.assertEqual(result.source, "PMII_W001.json")
            assert result.path is not None
            stored = json.loads(result.path.read_text(encoding="utf-8"))
            self.assertEqual(stored["worker_id"], "PMII / W001")
            self.assertEqual(set(stored), {"worker_id", "name", "face_image"})
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_duplicate_id_uses_internal_json_not_filename_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unrelated-name.json").write_text(
                json.dumps(_profile("W001", name="Original")), encoding="utf-8"
            )
            repository = WorkerProfileRepository(root)

            result = repository.create(_profile("W001", name="Replacement"))

            self.assertFalse(result.success)
            self.assertEqual(result.errors[0].code, "DUPLICATE_WORKER_ID")
            original = json.loads((root / "unrelated-name.json").read_text(encoding="utf-8"))
            self.assertEqual(original["name"], "Original")

    def test_create_avoids_sanitized_filename_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = WorkerProfileRepository(root)

            first = repository.create(_profile("A/B"))
            second = repository.create(_profile("A?B"))

            self.assertTrue(first.success)
            self.assertTrue(second.success)
            self.assertEqual(first.source, "A_B.json")
            self.assertEqual(second.source, "A_B_2.json")

    def test_edit_preserves_worker_id_and_atomically_saves_valid_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = WorkerProfileRepository(Path(directory))
            created = repository.create(_profile())
            assert created.source is not None

            changed_id = repository.edit(created.source, _profile("W002"))
            saved = repository.edit(
                created.source,
                _profile("W001", name="Updated Worker"),
            )

            self.assertFalse(changed_id.success)
            self.assertEqual(changed_id.errors[0].code, "IMMUTABLE_WORKER_ID")
            self.assertTrue(saved.success)
            loaded = repository.load(created.source)
            self.assertTrue(loaded.valid)
            assert loaded.profile is not None
            self.assertEqual(loaded.profile.worker_id, "W001")
            self.assertEqual(loaded.profile.name, "Updated Worker")

    def test_invalid_edit_does_not_change_existing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = WorkerProfileRepository(Path(directory))
            created = repository.create(_profile(name="Original"))
            assert created.source is not None

            result = repository.edit(
                created.source,
                _profile(name="", face_image="not-a-data-uri"),
            )

            self.assertFalse(result.success)
            loaded = repository.load(created.source)
            assert loaded.profile is not None
            self.assertEqual(loaded.profile.name, "Original")

    def test_list_includes_invalid_json_and_raw_read_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = WorkerProfileRepository(root)
            repository.create(_profile())
            (root / "broken.json").write_text("{bad", encoding="utf-8")

            records = repository.list_records()

            self.assertEqual(tuple(record.source for record in records), ("W001.json", "broken.json"))
            broken = next(record for record in records if record.source == "broken.json")
            self.assertFalse(broken.valid)
            self.assertEqual(broken.errors[0].code, "INVALID_WORKER_PROFILE_JSON")
            self.assertEqual(repository.read_raw("broken.json").raw_text, "{bad")

    def test_load_rejects_traversal_and_non_json_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = WorkerProfileRepository(Path(directory))

            for source in ("../profile.json", "sub/profile.json", "profile.txt", ".."):
                with self.subTest(source=source):
                    record = repository.load(source)
                    self.assertFalse(record.valid)
                    self.assertEqual(record.errors[0].code, "INVALID_PROFILE_SOURCE")

    def test_filename_sanitization_never_changes_profile_identity(self) -> None:
        self.assertEqual(sanitize_profile_filename("../../ W:001?*"), "W_001.json")
        self.assertEqual(sanitize_profile_filename("CON"), "_CON.json")
        self.assertEqual(sanitize_profile_filename("////"), "worker_profile.json")


if __name__ == "__main__":
    unittest.main()
