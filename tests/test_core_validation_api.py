from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

import signatus_core.app as core_app
from signatus_core.worksites import WorksiteRepository


def embedding(axis: int = 0) -> list[float]:
    values = [0.0] * 128
    values[axis] = 1.0
    return values


def write_worksite(
    path: Path,
    worksite_id: str,
    *,
    workers: list[object],
    required_ppe: list[str],
) -> None:
    path.write_text(
        json.dumps(
            {
                "worksite_id": worksite_id,
                "name": f"Gate {worksite_id}",
                "authorized_workers": workers,
                "required_ppe": required_ppe,
            }
        ),
        encoding="utf-8",
    )


class CoreValidationAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_data_errors_are_reported_and_smallest_scope_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_worksite(
                root / "WO-002.json",
                "WO-002",
                workers=[
                    {"worker_id": "GOOD", "name": "Good", "embedding": embedding()},
                    {"worker_id": "BAD", "name": "Bad", "embedding": [1.0, 0.0, 0.0]},
                ],
                required_ppe=["helmet"],
            )
            write_worksite(
                root / "WO-004.json",
                "WO-004",
                workers=[{"worker_id": "GOOD", "name": "Good", "embedding": embedding()}],
                required_ppe=["unapproved-item"],
            )
            repository = WorksiteRepository(root)

            with patch.object(core_app, "worksites", repository):
                report = await core_app.validation_report()
                worksites = await core_app.list_worksites()

        self.assertEqual(report["status"], "PASS_WITH_DATA_ERRORS")
        self.assertEqual(report["fatal_count"], 0)
        self.assertGreaterEqual(report["data_error_count"], 2)
        partial = next(item for item in worksites if item["worksite_id"] == "WO-002")
        broken = next(item for item in worksites if item["worksite_id"] == "WO-004")
        self.assertTrue(partial["available"])
        self.assertEqual(partial["valid_worker_count"], 1)
        self.assertEqual(partial["invalid_worker_count"], 1)
        self.assertFalse(broken["available"])

    async def test_unavailable_worksite_selection_conflicts_but_valid_one_selects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_worksite(
                root / "WO-GOOD.json",
                "WO-GOOD",
                workers=[{"worker_id": "GOOD", "name": "Good", "embedding": embedding()}],
                required_ppe=["helmet"],
            )
            write_worksite(
                root / "WO-BAD.json",
                "WO-BAD",
                workers=[{"worker_id": "GOOD", "name": "Good", "embedding": embedding()}],
                required_ppe=["unapproved-item"],
            )
            repository = WorksiteRepository(root)
            controller = SimpleNamespace(
                state=SimpleNamespace(value="STANDBY"),
                select_worksite=lambda _worksite, **_kwargs: None,
            )

            with (
                patch.object(core_app, "worksites", repository),
                patch.object(core_app, "controller", controller),
            ):
                selected = await core_app.select_worksite("WO-GOOD")
                with self.assertRaises(HTTPException) as raised:
                    await core_app.select_worksite("WO-BAD")

        self.assertEqual(selected["worksite_id"], "WO-GOOD")
        self.assertEqual(raised.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
