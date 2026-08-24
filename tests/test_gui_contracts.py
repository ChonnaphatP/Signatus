from __future__ import annotations

import unittest

from signatus_gui.contracts import (
    CameraState,
    OutcomeStatus,
    ValidationSeverity,
    parse_camera_status,
    parse_import_summary,
    parse_manager_catalog,
    parse_outcome,
    parse_raw_json_document,
    parse_validation_report,
    parse_worker_profile,
    parse_worksite_draft,
    parse_worksite_worker,
    parse_worksites,
)


class GUIContractTests(unittest.TestCase):
    def test_parses_worksites(self) -> None:
        result = parse_worksites(
            [
                {
                    "worksite_id": "WO-014",
                    "name": "North Gate",
                    "required_ppe": ["helmet", "gloves"],
                    "available": True,
                    "valid_worker_count": 48,
                    "invalid_worker_count": 2,
                }
            ]
        )
        self.assertEqual(result[0].worksite_id, "WO-014")
        self.assertEqual(result[0].required_ppe, ("helmet", "gloves"))
        self.assertTrue(result[0].available)
        self.assertEqual(result[0].invalid_worker_count, 2)

    def test_parses_camera_and_validation_status(self) -> None:
        camera = parse_camera_status({"state": "ERROR", "error": "device unavailable"})
        report = parse_validation_report(
            {
                "issues": [
                    {
                        "severity": "DATA_ERROR",
                        "code": "SFACE_DESCRIPTOR_LENGTH",
                        "message": "Expected 128 values, found 3.",
                        "worksite_id": "002",
                        "worker_id": "W017",
                    }
                ]
            }
        )

        self.assertEqual(camera.state, CameraState.ERROR)
        self.assertEqual(camera.error, "device unavailable")
        self.assertEqual(report.data_errors[0].severity, ValidationSeverity.DATA_ERROR)
        self.assertEqual(report.data_errors[0].worker_id, "W017")

    def test_rejects_invalid_worksite_availability_and_validation_severity(self) -> None:
        with self.assertRaisesRegex(ValueError, "availability"):
            parse_worksites([{"worksite_id": "002", "name": "Gate", "available": 1}])
        with self.assertRaisesRegex(ValueError, "severity"):
            parse_validation_report(
                {"issues": [{"severity": "BROKEN", "code": "X", "message": "bad"}]}
            )

    def test_parses_all_core_outcomes(self) -> None:
        payloads = [
            {"status": "AUTHORIZED", "worker_id": "EMP0017"},
            {
                "status": "PPE_VIOLATION",
                "worker_id": "EMP0017",
                "missing_ppe": ["gloves"],
            },
            {"status": "UNAUTHORIZED"},
            {
                "status": "FACE_CAPTURE_FAILED",
                "face_failure_reason": "NO_FACE",
                "attempt": 2,
                "retry_allowed": True,
            },
        ]
        results = [parse_outcome(payload) for payload in payloads]
        self.assertEqual([result.status for result in results], list(OutcomeStatus))
        self.assertEqual(results[1].missing_ppe, ("gloves",))

    def test_rejects_unknown_outcome(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown outcome"):
            parse_outcome({"status": "ALLOW"})

    def test_rejects_invalid_missing_ppe(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing-PPE"):
            parse_outcome({"status": "PPE_VIOLATION", "missing_ppe": "gloves"})

    def test_parses_manager_invalid_entry_and_editor_data(self) -> None:
        entries = parse_manager_catalog(
            [
                {
                    "source": "broken.json",
                    "worksite_id": None,
                    "name": None,
                    "required_ppe": [],
                    "available": False,
                    "unavailable_reason": "Invalid JSON",
                    "valid_worker_count": 0,
                    "invalid_worker_count": 0,
                    "issues": [
                        {
                            "severity": "DATA_ERROR",
                            "code": "INVALID_WORKSITE_JSON",
                            "message": "Invalid JSON",
                            "source": "broken.json",
                        }
                    ],
                    "active": False,
                }
            ]
        )
        draft = parse_worksite_draft(
            {
                "source": "WO-1.json",
                "worksite_id": "WO-1",
                "name": "Gate",
                "authorized_workers": [
                    {
                        "worker_id": "W1",
                        "name": "Worker One",
                        "embedding": [1.0] + [0.0] * 127,
                    }
                ],
                "required_ppe": [],
                "invalid_worker_messages": [],
                "active": True,
            }
        )

        self.assertFalse(entries[0].available)
        self.assertEqual(entries[0].source, "broken.json")
        self.assertEqual(draft.authorized_workers[0].name, "Worker One")
        self.assertTrue(draft.active)

    def test_parses_backend_import_results_and_malformed_json_view(self) -> None:
        summary = parse_import_summary(
            {
                "imported": [
                    {
                        "worksite_id": "WO-1",
                        "source": "download.json",
                        "message": "Imported.",
                        "skipped_workers": [
                            {"worker_id": "BAD", "message": "Invalid embedding."}
                        ],
                    }
                ],
                "skipped": [],
                "failed": [],
            }
        )
        document = parse_raw_json_document(
            {
                "source": "broken.json",
                "raw": "{broken",
                "formatted": None,
                "parse_error": "invalid JSON",
            }
        )

        self.assertIn("WO-1", summary.imported[0])
        self.assertIn("BAD", summary.worker_warnings[0])
        self.assertEqual(document.formatted, "{broken")

    def test_worker_profile_contains_face_image_but_no_embedding(self) -> None:
        profile = parse_worker_profile(
            {
                "worker_id": "W001",
                "name": "Worker One",
                "face_image": "data:image/jpeg;base64,AA==",
            }
        )

        self.assertEqual(profile.worker_id, "W001")
        self.assertEqual(profile.face_image, "data:image/jpeg;base64,AA==")
        self.assertFalse(hasattr(profile, "embedding"))

    def test_worksite_worker_requires_exactly_one_biometric_representation(self) -> None:
        stored = parse_worksite_worker(
            {
                "worker_id": "W001",
                "name": "Stored Worker",
                "embedding": [1.0] + [0.0] * 127,
            }
        )
        enrollment = parse_worksite_worker(
            {
                "worker_id": "W002",
                "name": "New Worker",
                "face_image": "data:image/jpeg;base64,AA==",
            }
        )

        self.assertIsNotNone(stored.embedding)
        self.assertIsNone(stored.face_image)
        self.assertIsNone(enrollment.embedding)
        self.assertIsNotNone(enrollment.face_image)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_worksite_worker(
                {
                    "worker_id": "W003",
                    "name": "Ambiguous Worker",
                    "embedding": [1.0] + [0.0] * 127,
                    "face_image": "data:image/jpeg;base64,AA==",
                }
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_worksite_worker({"worker_id": "W004", "name": "Missing biometric"})


if __name__ == "__main__":
    unittest.main()
