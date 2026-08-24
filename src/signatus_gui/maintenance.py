from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from .contracts import (
    ImportSummary,
    ManagedWorksite,
    WorkerProfile,
    parse_worksite_draft,
)
from .worksite_views import (
    ImportSummaryDialog,
    JsonViewerDialog,
    UrlImportDialog,
    WorkerProfileEditor,
    WorksiteDetailsDialog,
    WorksiteForm,
)

if TYPE_CHECKING:
    from .client import CoreClient
    from .views import MainWindow


class MaintenanceCoordinator(QObject):
    """Connect maintenance UI to Core without touching runtime selection."""

    def __init__(self, window: MainWindow, client: CoreClient) -> None:
        super().__init__(window)
        self._window = window
        self._client = client
        self._ppe_options: tuple[str, ...] = ()
        self._options_loaded = False
        self._pending_create = False
        self._pending_edit: ManagedWorksite | None = None
        self._pending_draft: object | None = None
        self._details_entry: ManagedWorksite | None = None
        self._worker_target_form: WorksiteForm | None = None
        self._profile_dialog: WorkerProfileEditor | None = None
        self._profile_source: str | None = None
        self._profile_open_pending = False
        self._pending_profile_source: str | None = None
        self._local_read_failures: tuple[str, ...] = ()
        self._pending_select_source: str | None = None
        self._dialogs: set[QDialog] = set()

        manager = window.manager
        window.manager_opened.connect(self.open_manager)
        manager.refresh_requested.connect(lambda: client.load_manager(refresh=True))
        manager.create_requested.connect(self._request_create)
        manager.edit_requested.connect(self._request_edit)
        manager.details_requested.connect(self._request_details)
        manager.view_json_requested.connect(
            lambda entry: client.load_manager_json(entry.source)
        )
        manager.delete_requested.connect(
            lambda entry: client.delete_worksite(entry.source)
        )
        manager.import_files_requested.connect(self._import_files)
        manager.import_folder_requested.connect(self._import_folder)
        manager.import_url_requested.connect(self._import_url)
        manager.worker_profile_tool_requested.connect(self._open_profile_creator)

        client.manager_catalog_loaded.connect(self._catalog_received)
        client.manager_options_loaded.connect(self._options_received)
        client.manager_draft_loaded.connect(self._draft_received)
        client.manager_details_loaded.connect(self._details_received)
        client.manager_json_loaded.connect(
            lambda document: self._open_dialog(JsonViewerDialog(document, window))
        )
        client.manager_changed.connect(self._manager_changed)
        client.manager_import_completed.connect(self._import_completed)
        client.management_failed.connect(self._management_failed)
        client.worker_profile_validated.connect(self._worker_profile_received)
        client.worker_profile_saved.connect(self._profile_saved)

    def open_manager(self) -> None:
        self._client.load_manager(refresh=True)
        if not self._options_loaded:
            self._client.load_manager_options()

    def _catalog_received(self, entries: tuple[ManagedWorksite, ...]) -> None:
        self._window.manager.set_entries(entries)
        if self._pending_select_source is not None and self._window.manager.select_source(
            self._pending_select_source
        ):
            self._pending_select_source = None
        self._client.load_worksites()

    def _request_create(self) -> None:
        self._pending_create = True
        self._pending_edit = None
        self._pending_draft = None
        if self._options_loaded:
            self._open_form()
        else:
            self._client.load_manager_options()

    def _request_edit(self, entry: ManagedWorksite) -> None:
        if not entry.available:
            self._window.manager.show_error(
                "Invalid Wo.No. files can be inspected or deleted but not structurally edited."
            )
            return
        self._pending_create = False
        self._pending_edit = entry
        self._pending_draft = None
        self._client.load_manager_draft(entry.source)
        if not self._options_loaded:
            self._client.load_manager_options()

    def _options_received(self, options: tuple[str, ...]) -> None:
        self._ppe_options = options
        self._options_loaded = True
        self._open_form_if_ready()

    def _draft_received(self, draft: object) -> None:
        self._pending_draft = draft
        self._open_form_if_ready()

    def _open_form_if_ready(self) -> None:
        if not self._options_loaded:
            return
        if self._pending_create:
            self._open_form()
        elif self._pending_edit is not None and self._pending_draft is not None:
            self._open_form(self._pending_draft)

    def _open_form(self, draft: object | None = None) -> None:
        form = WorksiteForm(self._ppe_options, draft=draft)
        form.cancel_requested.connect(self._window.close_worksite_form)
        form.worker_profile_requested.connect(lambda: self._choose_worker_profile(form))
        form.save_requested.connect(lambda payload: self._save_form(form, payload))
        self._window.show_worksite_form(form)
        self._pending_create = False
        self._pending_edit = None
        self._pending_draft = None

    def _save_form(self, form: WorksiteForm, payload: dict[str, object]) -> None:
        if form.source is None:
            self._client.create_worksite(payload)
        else:
            self._client.edit_worksite(form.source, payload)

    def _manager_changed(self, result: object) -> None:
        message = "Wo.No. maintenance completed."
        active_unchanged = False
        if isinstance(result, dict):
            if isinstance(result.get("message"), str):
                message = result["message"]
            active_unchanged = result.get("active_policy_unchanged") is True
        self._window.close_worksite_form()
        self._client.load_manager(refresh=False)
        self._client.load_worksites()
        if active_unchanged:
            message += "\n\nThe active Core policy remains unchanged until normal reselection."
        QMessageBox.information(self._window, "Wo.No. Manager", message)

    def _request_details(self, entry: ManagedWorksite) -> None:
        self._details_entry = entry
        self._client.load_manager_details(entry.source)

    def _details_received(self, payload: object) -> None:
        entry = self._details_entry
        self._details_entry = None
        try:
            draft = parse_worksite_draft(payload)
        except ValueError:
            dialog = WorksiteDetailsDialog(entry=entry, parent=self._window)
        else:
            dialog = WorksiteDetailsDialog(entry=entry, draft=draft, parent=self._window)
        self._open_dialog(dialog)

    def _import_files(self) -> None:
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self._window,
            "Import Wo.No. JSON files",
            "",
            "JSON files (*.json);;All files (*)",
        )
        self._import_local_paths(tuple(Path(path) for path in paths))

    def _import_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self._window,
            "Import Wo.No. folder",
        )
        if not selected:
            return
        try:
            paths = tuple(sorted(Path(selected).glob("*.json")))
        except OSError as error:
            self._show_failure(f"The selected folder cannot be scanned: {error}")
            return
        self._import_local_paths(paths)

    def _import_local_paths(self, paths: tuple[Path, ...]) -> None:
        if not paths:
            return
        documents: list[dict[str, str]] = []
        failures: list[str] = []
        for path in paths:
            try:
                documents.append(
                    {"source_name": path.name, "content": path.read_text(encoding="utf-8")}
                )
            except (OSError, UnicodeError) as error:
                failures.append(f"{path.name}: {error}")
        self._local_read_failures = tuple(failures)
        if documents:
            self._client.import_worksite_documents(documents)
        else:
            self._import_completed(ImportSummary(failed=self._local_read_failures))

    def _import_url(self) -> None:
        dialog = UrlImportDialog(self._window)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._client.import_worksite_url(dialog.url())

    def _import_completed(self, summary: ImportSummary) -> None:
        if self._local_read_failures:
            summary = replace(
                summary,
                failed=(*summary.failed, *self._local_read_failures),
            )
            self._local_read_failures = ()
        self._open_dialog(ImportSummaryDialog(summary, self._window))
        self._client.load_manager(refresh=False)
        self._client.load_worksites()

    def _choose_worker_profile(self, form: WorksiteForm) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self._window,
            "Add Worker Profile",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            form.show_error(f"Worker Profile cannot be loaded: {error}")
            return
        self._worker_target_form = form
        self._client.validate_worker_profile(payload)

    def _worker_profile_received(self, profile: WorkerProfile) -> None:
        if self._worker_target_form is not None:
            self._worker_target_form.add_worker_profile(profile)
            self._worker_target_form = None
        elif self._profile_open_pending:
            self._profile_open_pending = False
            self._profile_source = self._pending_profile_source
            self._pending_profile_source = None
            self._show_profile_editor(profile)

    def _open_profile_creator(self) -> None:
        dialog = QMessageBox(self._window)
        dialog.setWindowTitle("Worker Profile Tool")
        dialog.setText("Create a new Worker Profile or open a stored profile for editing.")
        create_button = dialog.addButton("Create New", QMessageBox.ButtonRole.AcceptRole)
        open_button = dialog.addButton("Open Existing", QMessageBox.ButtonRole.ActionRole)
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        if dialog.clickedButton() is create_button:
            self._profile_source = None
            self._show_profile_editor(None)
        elif dialog.clickedButton() is open_button:
            self._choose_profile_to_edit()

    def _choose_profile_to_edit(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self._window,
            "Open Worker Profile",
            "config/worker_profiles",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            self._show_failure(f"Worker Profile cannot be loaded: {error}")
            return
        self._profile_open_pending = True
        self._pending_profile_source = Path(path).name
        self._client.validate_worker_profile(payload)

    def _show_profile_editor(self, profile: WorkerProfile | None) -> None:
        dialog = WorkerProfileEditor(profile, self._window)
        dialog.save_requested.connect(self._save_profile)
        dialog.finished.connect(lambda _result: self._clear_profile_dialog(dialog))
        self._profile_dialog = dialog
        dialog.open()

    def _save_profile(self, profile: WorkerProfile) -> None:
        payload = {
            "worker_id": profile.worker_id,
            "name": profile.name,
            "face_image": profile.face_image,
        }
        if self._profile_source is None:
            self._client.create_worker_profile(payload)
        else:
            self._client.edit_worker_profile(self._profile_source, payload)

    def _profile_saved(self, profile: WorkerProfile) -> None:
        if self._profile_dialog is not None:
            self._profile_dialog.accept()
        QMessageBox.information(
            self._window,
            "Worker Profile",
            f"Saved Worker Profile {profile.worker_id}.",
        )

    def _clear_profile_dialog(self, dialog: WorkerProfileEditor) -> None:
        if self._profile_dialog is dialog:
            self._profile_dialog = None
            self._profile_source = None
            self._pending_profile_source = None
            self._profile_open_pending = False

    def _management_failed(self, error: object) -> None:
        message = str(error)
        detail: object = None
        if isinstance(error, dict):
            message = str(error.get("message", "Maintenance request failed."))
            detail = error.get("detail")
        if isinstance(detail, dict) and detail.get("code") == "DUPLICATE_WORKSITE_ID":
            existing = detail.get("existing_source")
            if isinstance(existing, str):
                self._offer_open_existing(existing, message)
                return
        if self._window._worksite_form is not None:
            self._window._worksite_form.show_error(message)
        else:
            self._window.manager.show_error(message)
        self._show_failure(message)

    def _offer_open_existing(self, source: str, message: str) -> None:
        dialog = QMessageBox(self._window)
        dialog.setWindowTitle("Wo.No. already exists")
        dialog.setText(message)
        open_button = dialog.addButton("Open Existing", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        if dialog.clickedButton() is open_button:
            self._window.close_worksite_form()
            if not self._window.manager.select_source(source):
                self._pending_select_source = source
                self._client.load_manager(refresh=True)

    def _show_failure(self, message: str) -> None:
        QMessageBox.warning(self._window, "Wo.No. Manager", message)

    def _open_dialog(self, dialog: QDialog) -> None:
        self._dialogs.add(dialog)
        dialog.finished.connect(lambda _result: self._dialogs.discard(dialog))
        dialog.open()
