from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import signatus_core.app as core_app
from signatus_core.domain import EmbeddingResult, FaceEmbeddingStatus
from signatus_core.worker_profiles import WorkerProfileRepository
from signatus_core.worksite_management import WorksiteManagementService
from signatus_core.worksites import WorksiteRepository


def embedding(axis: int = 0) -> list[float]:
    values = [0.0] * 128
    values[axis] = 1.0
    return values


def worksite(worksite_id: str, name: str = "Gate") -> dict[str, object]:
    return {
        "worksite_id": worksite_id,
        "name": name,
        "authorized_workers": [
            {
                "worker_id": "W001",
                "name": "Worker One",
                "embedding": embedding(),
            }
        ],
        "required_ppe": [],
    }


class CoreWorksiteManagerAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_manager_create_generates_and_stores_profile_embedding(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"profile-image"
        face_image = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        payload = {
            "worksite_id": "WO-ENROLL",
            "name": "Enrollment Gate",
            "authorized_workers": [
                {
                    "worker_id": "W001",
                    "name": "Worker One",
                    "face_image": face_image,
                }
            ],
            "required_ppe": [],
        }
        generate = AsyncMock(
            return_value=EmbeddingResult(
                track_id=0,
                status=FaceEmbeddingStatus.OK,
                embedding=embedding(),
            )
        )
        ai_client = SimpleNamespace(generate_profile_embedding=generate)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = WorksiteRepository(root)
            service = WorksiteManagementService(root, repository)
            controller = SimpleNamespace(
                selected_worksite=None,
                selected_worksite_source=None,
            )
            with (
                patch.object(core_app, "worksite_manager", service),
                patch.object(core_app, "controller", controller),
                patch.object(core_app, "ai_client", ai_client),
            ):
                result = await core_app.manager_create(payload)
            stored = json.loads((root / result["source"]).read_text(encoding="utf-8"))

        generate.assert_awaited_once_with(face_image)
        self.assertEqual(controller.selected_worksite, None)
        stored_worker = stored["authorized_workers"][0]
        self.assertEqual(set(stored_worker), {"worker_id", "name", "embedding"})
        self.assertEqual(len(stored_worker["embedding"]), 128)
        self.assertNotIn("face_image", stored_worker)

    async def test_manager_lists_malformed_file_but_runtime_selector_omits_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.json").write_text("{broken", encoding="utf-8")
            (root / "valid.json").write_text(
                json.dumps(worksite("WO-1")), encoding="utf-8"
            )
            repository = WorksiteRepository(root)
            service = WorksiteManagementService(root, repository)
            controller = SimpleNamespace(
                selected_worksite=None,
                selected_worksite_source=None,
            )

            with (
                patch.object(core_app, "worksites", repository),
                patch.object(core_app, "worksite_manager", service),
                patch.object(core_app, "controller", controller),
            ):
                managed = await core_app.manager_catalog()
                runtime = await core_app.list_worksites()

        self.assertEqual({entry["source"] for entry in managed}, {"broken.json", "valid.json"})
        broken = next(entry for entry in managed if entry["source"] == "broken.json")
        self.assertFalse(broken["available"])
        self.assertEqual([entry["worksite_id"] for entry in runtime], ["WO-1"])

    async def test_refresh_and_active_edit_keep_the_selected_policy_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "odd-name.json"
            source.write_text(json.dumps(worksite("WO-1", "Old policy")), encoding="utf-8")
            repository = WorksiteRepository(root)
            selected = repository.get("WO-1")
            assert selected is not None
            controller = SimpleNamespace(
                selected_worksite=selected,
                selected_worksite_source=source.name,
            )
            service = WorksiteManagementService(root, repository)
            edited = worksite("WO-1", "New disk policy")

            with (
                patch.object(core_app, "worksites", repository),
                patch.object(core_app, "worksite_manager", service),
                patch.object(core_app, "controller", controller),
            ):
                await core_app.refresh_manager_catalog()
                result = await core_app.manager_edit(source.name, edited)
                disk_name = json.loads(source.read_text())["name"]

        self.assertIs(controller.selected_worksite, selected)
        self.assertEqual(controller.selected_worksite.name, "Old policy")
        self.assertTrue(result["active_policy_unchanged"])
        self.assertEqual(disk_name, "New disk policy")

    async def test_active_edit_embeds_only_new_profile_worker_on_disk(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"new-profile-image"
        face_image = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        generate = AsyncMock(
            return_value=EmbeddingResult(
                track_id=0,
                status=FaceEmbeddingStatus.OK,
                embedding=embedding(1),
            )
        )
        ai_client = SimpleNamespace(generate_profile_embedding=generate)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "active.json"
            source.write_text(json.dumps(worksite("WO-1", "Old policy")), encoding="utf-8")
            repository = WorksiteRepository(root)
            selected = repository.get("WO-1")
            assert selected is not None
            controller = SimpleNamespace(
                selected_worksite=selected,
                selected_worksite_source=source.name,
            )
            service = WorksiteManagementService(root, repository)
            edited = worksite("WO-1", "New disk policy")
            edited["authorized_workers"].append(
                {
                    "worker_id": "W002",
                    "name": "Worker Two",
                    "face_image": face_image,
                }
            )

            with (
                patch.object(core_app, "worksite_manager", service),
                patch.object(core_app, "controller", controller),
                patch.object(core_app, "ai_client", ai_client),
            ):
                result = await core_app.manager_edit(source.name, edited)
            stored = json.loads(source.read_text(encoding="utf-8"))

        generate.assert_awaited_once_with(face_image)
        self.assertIs(controller.selected_worksite, selected)
        self.assertEqual(controller.selected_worksite.name, "Old policy")
        self.assertTrue(result["active_policy_unchanged"])
        self.assertEqual(len(stored["authorized_workers"]), 2)
        self.assertTrue(
            all("face_image" not in worker for worker in stored["authorized_workers"])
        )

    async def test_create_and_import_never_call_runtime_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = WorksiteRepository(root)
            selected_calls: list[object] = []
            controller = SimpleNamespace(
                selected_worksite=None,
                selected_worksite_source=None,
                select_worksite=lambda value, **_kwargs: selected_calls.append(value),
            )
            service = WorksiteManagementService(root, repository)

            with (
                patch.object(core_app, "worksites", repository),
                patch.object(core_app, "worksite_manager", service),
                patch.object(core_app, "controller", controller),
            ):
                await core_app.manager_create(worksite("WO-CREATED"))
                imported = await core_app.manager_import_local(
                    {
                        "documents": [
                            {
                                "source_name": "external.json",
                                "content": json.dumps(worksite("WO-IMPORTED")),
                            }
                        ]
                    }
                )

        self.assertEqual(selected_calls, [])
        self.assertEqual(imported["imported_count"], 1)

    async def test_active_delete_is_refused_even_when_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "active.json"
            source.write_text(json.dumps(worksite("WO-ACTIVE")), encoding="utf-8")
            repository = WorksiteRepository(root)
            active = repository.get("WO-ACTIVE")
            controller = SimpleNamespace(
                selected_worksite=active,
                selected_worksite_source=source.name,
            )
            service = WorksiteManagementService(root, repository)

            with (
                patch.object(core_app, "worksite_manager", service),
                patch.object(core_app, "controller", controller),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await core_app.manager_delete(source.name, confirmed=True)
                source_still_exists = source.exists()

        self.assertEqual(raised.exception.status_code, 409)
        self.assertTrue(source_still_exists)

    async def test_duplicate_create_reports_existing_source_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "WO-1.synthetic-valid.json"
            existing.write_text(json.dumps(worksite("WO-1")), encoding="utf-8")
            repository = WorksiteRepository(root)
            service = WorksiteManagementService(root, repository)
            controller = SimpleNamespace(
                selected_worksite=None,
                selected_worksite_source=None,
            )

            with (
                patch.object(core_app, "worksite_manager", service),
                patch.object(core_app, "controller", controller),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await core_app.manager_create(worksite("WO-1", "Duplicate"))
                existing_name = json.loads(existing.read_text())["name"]

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.detail["existing_source"], existing.name)
        self.assertEqual(existing_name, "Gate")

    async def test_worker_profile_endpoints_validate_create_and_edit(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"profile-image"
        profile = {
            "worker_id": "W001",
            "name": "Worker One",
            "face_image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        }
        with tempfile.TemporaryDirectory() as directory:
            repository = WorkerProfileRepository(Path(directory))
            with patch.object(core_app, "worker_profiles", repository):
                validated = await core_app.validate_profile(profile)
                created = await core_app.create_profile(profile)
                edited_payload = {**profile, "name": "Updated Worker"}
                edited = await core_app.edit_profile(created["source"], edited_payload)

        self.assertEqual(validated["worker_id"], "W001")
        self.assertTrue(created["success"])
        self.assertEqual(edited["profile"]["name"], "Updated Worker")


if __name__ == "__main__":
    unittest.main()
