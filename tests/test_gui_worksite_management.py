from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from signatus_gui.contracts import (
        CameraState,
        CameraStatus,
        ImportSummary,
        ManagedWorksite,
        RawJsonDocument,
        WorkerProfile,
        Worksite,
        WorksiteDraft,
        WorksiteWorker,
    )
    from signatus_gui.maintenance import MaintenanceCoordinator
    from signatus_gui.views import MainWindow
    from signatus_gui.worksite_views import (
        ImportSummaryDialog,
        JsonViewerDialog,
        UrlImportDialog,
        WorkerProfileEditor,
        WorksiteForm,
        WorksiteManager,
    )

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


if PYSIDE_AVAILABLE:

    class FakeCoreClient(QObject):
        manager_catalog_loaded = Signal(object)
        manager_options_loaded = Signal(object)
        manager_draft_loaded = Signal(object)
        manager_details_loaded = Signal(object)
        manager_json_loaded = Signal(object)
        manager_changed = Signal(object)
        manager_import_completed = Signal(object)
        management_failed = Signal(object)
        worker_profile_validated = Signal(object)
        worker_profile_saved = Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self.imported_documents: list[dict[str, str]] = []
            self.created_profile: dict[str, object] | None = None

        def import_worksite_documents(self, documents: list[dict[str, str]]) -> None:
            self.imported_documents = documents

        def load_manager(self, *, refresh: bool = False) -> None:
            del refresh

        def load_manager_options(self) -> None:
            pass

        def load_worksites(self) -> None:
            pass

        def create_worker_profile(self, payload: dict[str, object]) -> None:
            self.created_profile = payload


def _embedding() -> tuple[float, ...]:
    return (1.0,) + (0.0,) * 127


def _worker(worker_id: str = "W001") -> WorksiteWorker:
    return WorksiteWorker(worker_id, f"Worker {worker_id}", _embedding())


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 GUI extra is not installed")
class WorksiteManagementViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_manager_row_selection_has_no_runtime_or_maintenance_side_effect(self) -> None:
        manager = WorksiteManager()
        entry = ManagedWorksite(
            source="/config/worksites/WO-001.json",
            worksite_id="WO-001",
            name="North Gate",
            required_ppe=("helmet",),
            available=True,
            unavailable_reason=None,
            valid_worker_count=1,
            invalid_worker_count=0,
        )
        emitted: list[str] = []
        manager.refresh_requested.connect(lambda: emitted.append("refresh"))
        manager.edit_requested.connect(lambda _entry: emitted.append("edit"))
        manager.details_requested.connect(lambda _entry: emitted.append("details"))
        manager.view_json_requested.connect(lambda _entry: emitted.append("json"))
        manager.delete_requested.connect(lambda _entry: emitted.append("delete"))

        manager.set_entries((entry,))
        manager._table.selectRow(0)
        self.application.processEvents()

        self.assertEqual(emitted, [])
        self.assertIs(manager.selected_entry(), entry)
        self.assertFalse(hasattr(manager, "worksite_chosen"))
        self.assertFalse(hasattr(manager, "select_requested"))

    def test_invalid_manager_entry_remains_inspectable(self) -> None:
        manager = WorksiteManager()
        invalid = ManagedWorksite(
            source="/config/worksites/broken.json",
            worksite_id=None,
            name=None,
            required_ppe=(),
            available=False,
            unavailable_reason="Invalid JSON",
            valid_worker_count=0,
            invalid_worker_count=0,
        )
        viewed: list[ManagedWorksite] = []
        manager.view_json_requested.connect(viewed.append)
        manager.set_entries((invalid,))
        manager._table.selectRow(0)

        manager._json_button.click()

        self.assertEqual(viewed, [invalid])
        self.assertTrue(manager._delete_button.isEnabled())
        self.assertFalse(manager._edit_button.isEnabled())
        self.assertIn("Invalid JSON", manager._selection_details.text())

    def test_manager_return_has_a_dedicated_signal(self) -> None:
        manager = WorksiteManager()
        returned: list[bool] = []
        manager.return_requested.connect(lambda: returned.append(True))

        manager._return_button.click()

        self.assertEqual(returned, [True])

    def test_main_window_manager_return_preserves_detection_runtime_display(self) -> None:
        window = MainWindow()
        active = Worksite("WO-ACTIVE", "Active policy", ("helmet",))
        window.show_detection(active)
        window.detection.set_core_state("AUTHORIZATION")
        window.set_camera_status(CameraStatus(CameraState.RUNNING))

        window.open_manager()
        window.manager.set_entries(
            (
                ManagedWorksite(
                    source="other.json",
                    worksite_id="WO-OTHER",
                    name="Other policy",
                    required_ppe=(),
                    available=True,
                    unavailable_reason=None,
                    valid_worker_count=1,
                    invalid_worker_count=0,
                ),
            )
        )
        window.manager._table.selectRow(0)
        window.return_from_manager()

        self.assertIs(window.pages.currentWidget(), window.detection)
        self.assertEqual(window.detection._worksite_number.text(), "WO-ACTIVE")
        self.assertIn("AUTHORIZING", window.detection._core_state.text())
        self.assertIn("RUNNING", window.detection._camera_state.text())

    def test_folder_import_scans_only_direct_json_files(self) -> None:
        window = MainWindow()
        client = FakeCoreClient()
        coordinator = MaintenanceCoordinator(window, client)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.json").write_text('{"worksite_id":"ONE"}', encoding="utf-8")
            (root / "ignore.txt").write_text("not json", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "two.json").write_text('{"worksite_id":"TWO"}', encoding="utf-8")

            with patch(
                "signatus_gui.maintenance.QFileDialog.getExistingDirectory",
                return_value=str(root),
            ):
                coordinator._import_folder()

        self.assertEqual(
            [document["source_name"] for document in client.imported_documents],
            ["one.json"],
        )

    def test_manager_delete_requires_confirmation_and_blocks_active_entry(self) -> None:
        manager = WorksiteManager()
        entry = ManagedWorksite(
            source="/config/worksites/WO-001.json",
            worksite_id="WO-001",
            name="North Gate",
            required_ppe=(),
            available=True,
            unavailable_reason=None,
            valid_worker_count=1,
            invalid_worker_count=0,
        )
        deleted: list[ManagedWorksite] = []
        manager.delete_requested.connect(deleted.append)
        manager.set_entries((entry,))
        manager._table.selectRow(0)

        manager.set_delete_confirmation_handler(lambda _entry: False)
        manager._delete_button.click()
        self.assertEqual(deleted, [])
        manager.set_delete_confirmation_handler(lambda _entry: True)
        manager._delete_button.click()
        self.assertEqual(deleted, [entry])

        active = ManagedWorksite(
            source=entry.source,
            worksite_id=entry.worksite_id,
            name=entry.name,
            required_ppe=entry.required_ppe,
            available=entry.available,
            unavailable_reason=entry.unavailable_reason,
            valid_worker_count=entry.valid_worker_count,
            invalid_worker_count=entry.invalid_worker_count,
            active=True,
        )
        manager.set_entries((active,))
        manager._table.selectRow(0)
        manager._delete_button.click()
        self.assertEqual(deleted, [entry])
        self.assertIn("active Wo.No.", manager._error.text())

    def test_create_form_requires_identity_name_and_worker(self) -> None:
        form = WorksiteForm(("helmet", "gloves"))

        self.assertEqual(form.validation_error(), "Worksite ID is required.")
        form._id.setText("WO-001")
        self.assertEqual(form.validation_error(), "Worksite name is required.")
        form._name.setText("North Gate")
        self.assertIn("At least one", form.validation_error() or "")
        self.assertFalse(form.submit())

    def test_form_adds_rejects_duplicates_and_removes_workers(self) -> None:
        form = WorksiteForm(("helmet",))
        worker = _worker()

        self.assertTrue(form.add_worker(worker))
        self.assertFalse(form.add_worker(worker))
        self.assertIn("already", form._error.text())
        form._worker_table.selectRow(0)
        self.assertTrue(form.remove_selected_worker())

        self.assertEqual(form.workers, ())

    def test_form_zero_ppe_is_valid_and_payload_is_compact(self) -> None:
        form = WorksiteForm(("helmet", "gloves"))
        form._id.setText("PMII / WO 015")
        form._name.setText("Pump maintenance")
        form.add_worker(_worker())

        payload = form.payload()

        self.assertEqual(payload["required_ppe"], [])
        self.assertEqual(
            set(payload),
            {"worksite_id", "name", "authorized_workers", "required_ppe"},
        )
        self.assertEqual(len(payload["authorized_workers"][0]["embedding"]), 128)

    def test_added_worker_profile_sends_face_image_for_core_enrollment(self) -> None:
        form = WorksiteForm(("helmet",))
        form._id.setText("WO-NEW")
        form._name.setText("New enrollment")
        profile = WorkerProfile(
            worker_id="W002",
            name="New Worker",
            face_image="data:image/jpeg;base64,AA==",
        )

        self.assertTrue(form.add_worker_profile(profile))
        payload = form.payload()
        worker = payload["authorized_workers"][0]

        self.assertEqual(worker["face_image"], profile.face_image)
        self.assertNotIn("embedding", worker)

    def test_edit_form_locks_worksite_id_and_emits_valid_changes(self) -> None:
        draft = WorksiteDraft(
            worksite_id="WO-015",
            name="Old name",
            authorized_workers=(_worker(),),
            required_ppe=("helmet",),
            source="/config/worksites/odd-file-name.json",
            active=True,
        )
        form = WorksiteForm(("helmet", "gloves"), draft)
        saved: list[dict[str, object]] = []
        form.save_requested.connect(saved.append)

        form._name.setText("New name")
        form._ppe["helmet"].setChecked(False)
        form._ppe["gloves"].setChecked(True)
        self.assertTrue(form.submit())

        self.assertTrue(form._id.isReadOnly())
        self.assertEqual(saved[0]["worksite_id"], "WO-015")
        self.assertEqual(saved[0]["name"], "New name")
        self.assertEqual(saved[0]["required_ppe"], ["gloves"])
        saved_worker = saved[0]["authorized_workers"][0]
        self.assertEqual(len(saved_worker["embedding"]), 128)
        self.assertNotIn("face_image", saved_worker)
        self.assertTrue(form.active)

    def test_json_viewer_is_read_only_for_valid_and_malformed_data(self) -> None:
        valid_raw = '{"worksite_id":"WO-001"}'
        valid = JsonViewerDialog(
            RawJsonDocument(
                "WO-001.json",
                valid_raw,
                json.dumps(json.loads(valid_raw), indent=2),
            )
        )
        malformed = JsonViewerDialog(
            RawJsonDocument("broken.json", "{oops", "{oops", "line 1: invalid JSON")
        )

        self.assertTrue(valid._text.isReadOnly())
        self.assertIn("\n", valid._text.toPlainText())
        self.assertTrue(malformed._text.isReadOnly())
        self.assertEqual(malformed._text.toPlainText(), "{oops")
        self.assertIn("invalid JSON", malformed._parse_error.text())

    def test_import_summary_is_one_read_only_dialog(self) -> None:
        dialog = ImportSummaryDialog(
            ImportSummary(
                imported=("WO-001", "WO-002"),
                skipped=("WO-003, duplicate",),
                failed=("broken.json, invalid JSON",),
                worker_warnings=("WO-002 skipped worker W009",),
            )
        )

        self.assertTrue(dialog._text.isReadOnly())
        self.assertIn("WO-001", dialog._text.toPlainText())
        self.assertIn("duplicate", dialog._text.toPlainText())
        self.assertIn("skipped worker", dialog._text.toPlainText())

    def test_url_dialog_accepts_only_direct_http_or_https_locations(self) -> None:
        self.assertIsNone(UrlImportDialog.validation_error("https://example.test/wo.json"))
        self.assertIsNone(UrlImportDialog.validation_error("http://example.test/wo.json"))
        self.assertIsNotNone(UrlImportDialog.validation_error("file:///tmp/wo.json"))
        self.assertIsNotNone(UrlImportDialog.validation_error("ftp://example.test/wo.json"))
        self.assertIsNotNone(UrlImportDialog.validation_error("https://user:pw@example.test/x"))

    def test_worker_profile_editor_validates_data_and_locks_id_when_editing(self) -> None:
        profile = WorkerProfile(
            worker_id="W001",
            name="Example Worker",
            face_image="data:image/jpeg;base64,AA==",
        )
        editor = WorkerProfileEditor(profile)
        saved: list[WorkerProfile] = []
        editor.save_requested.connect(saved.append)

        editor._name.setText("Updated Worker")
        self.assertTrue(editor.submit())

        self.assertTrue(editor._id.isReadOnly())
        self.assertEqual(saved[0].worker_id, "W001")
        self.assertEqual(saved[0].name, "Updated Worker")
        self.assertFalse(hasattr(saved[0], "embedding"))
        self.assertFalse(hasattr(editor, "_embedding"))
        self.assertEqual(
            set(editor.payload()),
            {"worker_id", "name", "face_image"},
        )

    def test_maintenance_profile_save_payload_contains_no_embedding(self) -> None:
        window = MainWindow()
        client = FakeCoreClient()
        coordinator = MaintenanceCoordinator(window, client)
        profile = WorkerProfile(
            worker_id="W009",
            name="No Descriptor UI",
            face_image="data:image/png;base64,AA==",
        )

        coordinator._save_profile(profile)

        self.assertEqual(
            client.created_profile,
            {
                "worker_id": "W009",
                "name": "No Descriptor UI",
                "face_image": "data:image/png;base64,AA==",
            },
        )


if __name__ == "__main__":
    unittest.main()
