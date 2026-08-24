from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from signatus_contracts import ValidationSeverity
from signatus_core.worksites import WorksiteRepository


def _valid_embedding(axis: int = 0) -> list[float]:
    values = [0.0] * 128
    values[axis] = 1.0
    return values


def _worksite(
    worksite_id: str,
    *,
    workers: list[object] | None = None,
    required_ppe: object = None,
) -> dict[str, object]:
    return {
        "worksite_id": worksite_id,
        "name": f"Gate {worksite_id}",
        "authorized_workers": (
            [{"worker_id": "EMP1", "name": "Employee 1", "embedding": _valid_embedding()}]
            if workers is None
            else workers
        ),
        "required_ppe": ["helmet"] if required_ppe is None else required_ppe,
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class WorksiteRepositoryTests(unittest.TestCase):
    def test_loads_strict_sface_worker_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.json", _worksite("WO-001"))

            catalog = WorksiteRepository(root).load_catalog()
            worksite = catalog.get("WO-001")

        self.assertIsNotNone(worksite)
        assert worksite is not None
        self.assertEqual(worksite.authorized_workers[0].worker_id, "EMP1")
        self.assertEqual(len(worksite.authorized_workers[0].embedding), 128)
        self.assertEqual(catalog.data_errors, ())

    def test_three_value_embedding_is_data_error_and_zero_valid_workers_disable_worksite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root / "WO-001.json",
                _worksite(
                    "WO-001",
                    workers=[
                        {
                            "worker_id": "EMP1",
                            "name": "Employee 1",
                            "embedding": [1.0, 0.0, 0.0],
                        }
                    ],
                ),
            )

            catalog = WorksiteRepository(root).load_catalog()
            worksite = catalog.get("WO-001")

        self.assertIsNone(worksite)
        self.assertEqual(catalog.records[0].invalid_worker_count, 1)
        self.assertFalse(catalog.records[0].available)
        issue = next(
            issue
            for issue in catalog.data_errors
            if issue.code == "INVALID_SFACE_DESCRIPTOR_LENGTH"
        )
        self.assertEqual(issue.severity, ValidationSeverity.DATA_ERROR)
        self.assertEqual(issue.worker_id, "EMP1")
        self.assertIn("Expected 128 values, found 3", issue.message)
        self.assertFalse(catalog.has_fatal_errors)

    def test_mixed_workers_keep_worksite_and_valid_workers_available(self) -> None:
        workers = [
            {"worker_id": "GOOD1", "name": "Good 1", "embedding": _valid_embedding(0)},
            {"worker_id": "BAD", "name": "Bad", "embedding": [1.0, 0.0, 0.0]},
            {"worker_id": "GOOD2", "name": "Good 2", "embedding": _valid_embedding(1)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.json", _worksite("WO-001", workers=workers))

            repository = WorksiteRepository(root)
            worksite = repository.get("WO-001")
            record = repository.get_record("WO-001")

        self.assertIsNotNone(worksite)
        self.assertIsNotNone(record)
        assert worksite is not None and record is not None
        self.assertEqual(
            tuple(worker.worker_id for worker in worksite.authorized_workers),
            ("GOOD1", "GOOD2"),
        )
        self.assertTrue(record.available)
        self.assertEqual(record.valid_worker_count, 2)
        self.assertEqual(record.invalid_worker_count, 1)

    def test_duplicate_worker_ids_disable_all_duplicates_only(self) -> None:
        workers = [
            {"worker_id": "DUP", "name": "Duplicate 1", "embedding": _valid_embedding(0)},
            {"worker_id": "DUP", "name": "Duplicate 2", "embedding": _valid_embedding(1)},
            {"worker_id": "GOOD", "name": "Good", "embedding": _valid_embedding(2)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.json", _worksite("WO-001", workers=workers))

            catalog = WorksiteRepository(root).load_catalog()
            worksite = catalog.get("WO-001")

        self.assertIsNotNone(worksite)
        assert worksite is not None
        self.assertEqual(
            tuple(worker.worker_id for worker in worksite.authorized_workers), ("GOOD",)
        )
        self.assertEqual(catalog.records[0].invalid_worker_count, 2)
        duplicate_issues = [
            issue for issue in catalog.data_errors if issue.code == "DUPLICATE_WORKER_ID"
        ]
        self.assertEqual(len(duplicate_issues), 2)

    def test_nonfinite_wrong_type_and_zero_embeddings_are_filtered(self) -> None:
        nonfinite = _valid_embedding()
        nonfinite[4] = math.inf
        wrong_type: list[object] = _valid_embedding()
        wrong_type[4] = "0.2"
        workers = [
            {"worker_id": "INF", "name": "Infinity", "embedding": nonfinite},
            {"worker_id": "TYPE", "name": "Wrong type", "embedding": wrong_type},
            {"worker_id": "ZERO", "name": "Zero", "embedding": [0.0] * 128},
            {"worker_id": "GOOD", "name": "Good", "embedding": _valid_embedding(3)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.json", _worksite("WO-001", workers=workers))

            catalog = WorksiteRepository(root).load_catalog()
            worksite = catalog.get("WO-001")

        self.assertIsNotNone(worksite)
        assert worksite is not None
        self.assertEqual(
            tuple(worker.worker_id for worker in worksite.authorized_workers), ("GOOD",)
        )
        self.assertEqual(
            {issue.code for issue in catalog.data_errors},
            {
                "NONFINITE_SFACE_DESCRIPTOR",
                "INVALID_SFACE_DESCRIPTOR_TYPE",
                "UNUSABLE_SFACE_DESCRIPTOR",
            },
        )

    def test_missing_embedding_and_nonobject_worker_do_not_disable_worksite(self) -> None:
        workers = [
            {"worker_id": "MISSING", "name": "Missing embedding"},
            "not-a-worker-object",
            {"worker_id": "GOOD", "name": "Good", "embedding": _valid_embedding(5)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.json", _worksite("WO-001", workers=workers))

            catalog = WorksiteRepository(root).load_catalog()
            worksite = catalog.get("WO-001")

        self.assertIsNotNone(worksite)
        assert worksite is not None
        self.assertEqual(
            tuple(worker.worker_id for worker in worksite.authorized_workers), ("GOOD",)
        )
        self.assertEqual(catalog.records[0].invalid_worker_count, 2)
        self.assertEqual(
            {issue.code for issue in catalog.data_errors},
            {"MISSING_SFACE_DESCRIPTOR", "INVALID_WORKER_RECORD"},
        )

    def test_invalid_ppe_disables_only_affected_worksite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root / "WO-BAD.json",
                _worksite("WO-BAD", required_ppe=["respirator"]),
            )
            _write(root / "WO-GOOD.json", _worksite("WO-GOOD"))

            catalog = WorksiteRepository(root).load_catalog()

        self.assertIsNone(catalog.get("WO-BAD"))
        self.assertIsNotNone(catalog.get("WO-GOOD"))
        bad_record = catalog.get_record("WO-BAD")
        self.assertIsNotNone(bad_record)
        assert bad_record is not None
        self.assertFalse(bad_record.available)
        self.assertIn("no approved Core policy", bad_record.unavailable_reason or "")
        self.assertFalse(catalog.has_fatal_errors)

    def test_invalid_json_does_not_hide_other_worksites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "WO-BAD.json").write_text("{bad", encoding="utf-8")
            _write(root / "WO-GOOD.json", _worksite("WO-GOOD"))

            catalog = WorksiteRepository(root).load_catalog()

        self.assertEqual(
            tuple(worksite.worksite_id for worksite in catalog.available_worksites),
            ("WO-GOOD",),
        )
        self.assertTrue(any(issue.code == "INVALID_WORKSITE_JSON" for issue in catalog.data_errors))

    def test_duplicate_worksite_ids_are_both_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "first.json", _worksite("WO-DUP"))
            _write(root / "second.json", _worksite("WO-DUP"))

            catalog = WorksiteRepository(root).load_catalog()

        self.assertEqual(catalog.available_worksites, ())
        self.assertEqual(len(catalog.records), 2)
        self.assertTrue(all(not record.available for record in catalog.records))
        self.assertEqual(
            len(
                [
                    issue
                    for issue in catalog.data_errors
                    if issue.code == "DUPLICATE_WORKSITE_ID"
                ]
            ),
            2,
        )
        self.assertFalse(catalog.has_fatal_errors)

    def test_empty_worksite_directory_is_warning_only_safe_standby(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = WorksiteRepository(Path(directory)).load_catalog()

        self.assertEqual(catalog.available_worksites, ())
        self.assertEqual(catalog.data_errors, ())
        self.assertEqual(
            tuple(issue.code for issue in catalog.warnings),
            ("NO_WORKSITE_CONFIGURATIONS",),
        )
        self.assertFalse(catalog.has_fatal_errors)

    def test_all_invalid_worksites_are_nonfatal_safe_standby(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-BAD.json", _worksite("WO-BAD", required_ppe=["unknown"]))

            catalog = WorksiteRepository(root).load_catalog()

        self.assertEqual(catalog.available_worksites, ())
        self.assertTrue(catalog.data_errors)
        self.assertFalse(catalog.has_fatal_errors)

    def test_example_file_is_reported_and_unavailable_for_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.example.json", _worksite("WO-001"))

            catalog = WorksiteRepository(root).load_catalog()

        self.assertIsNone(catalog.get("WO-001"))
        self.assertFalse(catalog.records[0].available)
        issue = next(
            issue
            for issue in catalog.data_errors
            if issue.code == "EXAMPLE_WORKSITE_DATA"
        )
        self.assertEqual(issue.worksite_id, "WO-001")
        self.assertFalse(catalog.has_fatal_errors)

    def test_missing_directory_is_fatal_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"

            catalog = WorksiteRepository(missing).load_catalog()

        self.assertEqual(catalog.available_worksites, ())
        self.assertEqual(
            tuple(issue.code for issue in catalog.fatal_issues),
            ("WORKSITE_DIRECTORY_MISSING",),
        )

    def test_catalog_is_stable_after_first_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.json", _worksite("WO-001"))
            repository = WorksiteRepository(root)

            first = repository.load_catalog()
            _write(root / "WO-002.json", _worksite("WO-002"))
            second = repository.load_catalog()

        self.assertIs(first, second)
        self.assertEqual(tuple(item.worksite_id for item in second.available_worksites), ("WO-001",))

    def test_refresh_catalog_builds_new_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.json", _worksite("WO-001"))
            repository = WorksiteRepository(root)

            first = repository.load_catalog()
            _write(root / "WO-002.json", _worksite("WO-002"))
            refreshed = repository.refresh_catalog()

        self.assertIsNot(first, refreshed)
        self.assertEqual(
            tuple(item.worksite_id for item in refreshed.available_worksites),
            ("WO-001", "WO-002"),
        )

    def test_missing_worker_name_disables_only_that_worker(self) -> None:
        workers = [
            {"worker_id": "NO-NAME", "embedding": _valid_embedding(0)},
            {"worker_id": "GOOD", "name": "Good Worker", "embedding": _valid_embedding(1)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "WO-001.json", _worksite("WO-001", workers=workers))

            catalog = WorksiteRepository(root).load_catalog()
            worksite = catalog.get("WO-001")

        self.assertIsNotNone(worksite)
        assert worksite is not None
        self.assertEqual(tuple(worker.worker_id for worker in worksite.authorized_workers), ("GOOD",))
        self.assertTrue(any(issue.code == "INVALID_WORKER_NAME" for issue in catalog.data_errors))


if __name__ == "__main__":
    unittest.main()
