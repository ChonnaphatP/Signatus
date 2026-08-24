from __future__ import annotations

import base64
import mimetypes
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .contracts import (
    GUIProtocolError,
    ImportSummary,
    ManagedWorksite,
    RawJsonDocument,
    WorkerProfile,
    WorksiteDraft,
    WorksiteWorker,
    parse_worker_profile,
    parse_worksite_worker,
)


class WorksiteManager(QWidget):
    """Maintenance-only Wo.No. catalog view.

    Selecting a table row deliberately has no outward signal. Only explicit
    maintenance buttons emit requests, and this class has no activate/select
    signal or knowledge of the runtime selection endpoint.
    """

    refresh_requested = Signal()
    create_requested = Signal()
    import_files_requested = Signal()
    import_folder_requested = Signal()
    import_url_requested = Signal()
    edit_requested = Signal(object)
    details_requested = Signal(object)
    view_json_requested = Signal(object)
    delete_requested = Signal(object)
    return_requested = Signal()
    worker_profile_tool_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._entries: tuple[ManagedWorksite, ...] = ()
        self._delete_confirmation: Callable[[ManagedWorksite], bool] = (
            lambda entry: self.confirm_deletion(self, entry)
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)
        title = QLabel("Worksite Number Manager")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        toolbar = QHBoxLayout()
        self._create_button = self._button("Create Wo.No.", "createWorksiteButton")
        self._import_files_button = self._button("Import Local", "importWorksiteFilesButton")
        self._import_folder_button = self._button("Import Folder", "importWorksiteFolderButton")
        self._import_url_button = self._button("Import URL", "importWorksiteUrlButton")
        self._profile_button = self._button("Worker Profile Tool", "workerProfileToolButton")
        self._refresh_button = self._button("Refresh", "refreshWorksitesButton")
        for button in (
            self._create_button,
            self._import_files_button,
            self._import_folder_button,
            self._import_url_button,
            self._profile_button,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        toolbar.addWidget(self._refresh_button)
        root.addLayout(toolbar)

        body = QGridLayout()
        body.setHorizontalSpacing(16)
        available = QLabel("Available Wo.No.")
        available.setObjectName("fieldCaption")
        selected = QLabel("Selected Wo.No.")
        selected.setObjectName("fieldCaption")
        body.addWidget(available, 0, 0)
        body.addWidget(selected, 0, 1)

        self._table = QTableWidget(0, 4)
        self._table.setObjectName("managedWorksiteTable")
        self._table.setHorizontalHeaderLabels(("Wo.No.", "Name", "Workers", "State"))
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().hide()
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._show_selected)
        body.addWidget(self._table, 1, 0)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self._selection_details = QLabel("Select a Wo.No. to inspect it.")
        self._selection_details.setObjectName("managedWorksiteDetails")
        self._selection_details.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._selection_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._selection_details.setWordWrap(True)
        detail_layout.addWidget(self._selection_details, 1)

        self._edit_button = self._button("Edit", "editWorksiteButton")
        self._details_button = self._button("Details", "worksiteDetailsButton")
        self._json_button = self._button("View JSON", "viewWorksiteJsonButton")
        self._delete_button = self._button("Delete", "deleteWorksiteButton")
        self._return_button = self._button("Return", "returnFromWorksiteManagerButton")
        for button in (
            self._edit_button,
            self._details_button,
            self._json_button,
            self._delete_button,
        ):
            button.setEnabled(False)
            detail_layout.addWidget(button)
        detail_layout.addSpacing(8)
        detail_layout.addWidget(self._return_button)
        body.addWidget(detail_panel, 1, 1)
        body.setColumnStretch(0, 3)
        body.setColumnStretch(1, 2)
        root.addLayout(body, 1)

        self._error = QLabel()
        self._error.setObjectName("error")
        self._error.setWordWrap(True)
        self._error.hide()
        root.addWidget(self._error)

        self._refresh_button.clicked.connect(self.refresh_requested)
        self._create_button.clicked.connect(self.create_requested)
        self._import_files_button.clicked.connect(self.import_files_requested)
        self._import_folder_button.clicked.connect(self.import_folder_requested)
        self._import_url_button.clicked.connect(self.import_url_requested)
        self._profile_button.clicked.connect(self.worker_profile_tool_requested)
        self._edit_button.clicked.connect(
            lambda: self._emit_for_selected(self.edit_requested)
        )
        self._details_button.clicked.connect(
            lambda: self._emit_for_selected(self.details_requested)
        )
        self._json_button.clicked.connect(
            lambda: self._emit_for_selected(self.view_json_requested)
        )
        self._delete_button.clicked.connect(self._request_delete)
        self._return_button.clicked.connect(self.return_requested)

    @staticmethod
    def _button(text: str, object_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        return button

    @property
    def entries(self) -> tuple[ManagedWorksite, ...]:
        return self._entries

    def set_entries(self, entries: tuple[ManagedWorksite, ...]) -> None:
        selected = self.selected_entry()
        selected_source = selected.source if selected is not None else None
        self._entries = tuple(entries)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        selected_row = -1
        for row, entry in enumerate(self._entries):
            self._table.insertRow(row)
            identity = entry.worksite_id or "INVALID"
            name = entry.name or Path(entry.source).name
            worker_count = str(entry.valid_worker_count)
            if entry.invalid_worker_count:
                worker_count = f"{worker_count} ({entry.invalid_worker_count} skipped)"
            state = "Available" if entry.available else "Invalid"
            if entry.active:
                state += " · Active"
            values = (identity, name, worker_count, state)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, entry)
                if not entry.available:
                    item.setForeground(QBrush(QColor("#8a3b2b")))
                self._table.setItem(row, column, item)
            if entry.source == selected_source:
                selected_row = row
        if selected_row >= 0:
            self._table.selectRow(selected_row)
        else:
            self._table.clearSelection()
            self._show_selected()
        self._error.hide()

    def selected_entry(self) -> ManagedWorksite | None:
        row = self._table.currentRow()
        if not 0 <= row < len(self._entries):
            return None
        return self._entries[row]

    def select_source(self, source: str) -> bool:
        for row, entry in enumerate(self._entries):
            if entry.source == source:
                self._table.selectRow(row)
                return True
        return False

    def show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.setVisible(bool(message))

    def set_delete_confirmation_handler(
        self,
        handler: Callable[[ManagedWorksite], bool],
    ) -> None:
        """Override confirmation handling, primarily for host integration/tests."""

        self._delete_confirmation = handler

    def _show_selected(self) -> None:
        entry = self.selected_entry()
        has_entry = entry is not None
        self._details_button.setEnabled(has_entry)
        self._json_button.setEnabled(has_entry)
        self._delete_button.setEnabled(has_entry)
        self._edit_button.setEnabled(bool(entry and entry.worksite_id and entry.available))
        if entry is None:
            self._selection_details.setText("Select a Wo.No. to inspect it.")
            return
        reason = entry.unavailable_reason or "-"
        lines = [
            f"Worksite ID: {entry.worksite_id or 'Unavailable'}",
            f"Name: {entry.name or '-'}",
            f"Valid workers: {entry.valid_worker_count}",
            f"Invalid workers: {entry.invalid_worker_count}",
            f"Required PPE: {', '.join(entry.required_ppe) or 'None'}",
            f"State: {'Available' if entry.available else 'Invalid / unavailable'}",
            f"Active in Core: {'Yes' if entry.active else 'No'}",
            f"File: {entry.source}",
        ]
        if not entry.available:
            lines.append(f"Reason: {reason}")
        if entry.issues:
            lines.append("")
            lines.append("Validation issues:")
            lines.extend(f"• {issue.message}" for issue in entry.issues)
        self._selection_details.setText("\n".join(lines))

    def _emit_for_selected(self, signal: Signal) -> None:
        entry = self.selected_entry()
        if entry is not None:
            signal.emit(entry)

    def _request_delete(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        if entry.active:
            self.show_error(
                "The active Wo.No. cannot be deleted. Select another Wo.No. through "
                "the runtime selector first."
            )
            return
        if self._delete_confirmation(entry):
            self.show_error("")
            self.delete_requested.emit(entry)

    @staticmethod
    def confirm_deletion(parent: QWidget, entry: ManagedWorksite) -> bool:
        """Ask for explicit confirmation without performing any deletion itself."""

        identity = entry.worksite_id or Path(entry.source).name
        dialog = QMessageBox(parent)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Delete Wo.No.")
        dialog.setText(f"Permanently delete {identity}?")
        dialog.setInformativeText("This removes only the selected stored JSON file.")
        delete_button = dialog.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        return dialog.clickedButton() is delete_button


class WorksiteForm(QWidget):
    """Reusable Create/Edit form that produces the compact Wo.No. payload."""

    worker_profile_requested = Signal()
    save_requested = Signal(object)
    cancel_requested = Signal()

    def __init__(
        self,
        ppe_classes: tuple[str, ...] | list[str],
        draft: WorksiteDraft | None = None,
    ) -> None:
        super().__init__()
        self._ppe_classes = tuple(dict.fromkeys(ppe_classes))
        self._workers: list[WorksiteWorker] = []
        self._source: str | None = None
        self._active = False
        self._editing = draft is not None
        self._unknown_ppe: tuple[str, ...] = ()

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)
        self._title = QLabel("Edit Wo.No." if self._editing else "Create Wo.No.")
        self._title.setObjectName("pageTitle")
        root.addWidget(self._title)

        fields = QFormLayout()
        self._id = QLineEdit()
        self._id.setObjectName("worksiteIdInput")
        self._id.setPlaceholderText("Required; any numbering format")
        self._name = QLineEdit()
        self._name.setObjectName("worksiteNameInput")
        self._name.setPlaceholderText("Required")
        fields.addRow("Worksite ID:", self._id)
        fields.addRow("Name:", self._name)
        root.addLayout(fields)

        worker_group = QGroupBox("Authorized workers")
        worker_layout = QVBoxLayout(worker_group)
        self._worker_table = QTableWidget(0, 2)
        self._worker_table.setObjectName("worksiteWorkerTable")
        self._worker_table.setHorizontalHeaderLabels(("Worker ID", "Name"))
        self._worker_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._worker_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._worker_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._worker_table.verticalHeader().hide()
        self._worker_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        worker_layout.addWidget(self._worker_table)
        worker_actions = QHBoxLayout()
        self._add_worker = QPushButton("Add Worker Profile")
        self._add_worker.setObjectName("addWorkerProfileButton")
        self._remove_worker = QPushButton("Remove Worker")
        self._remove_worker.setObjectName("removeWorksiteWorkerButton")
        worker_actions.addWidget(self._add_worker)
        worker_actions.addWidget(self._remove_worker)
        worker_actions.addStretch(1)
        worker_layout.addLayout(worker_actions)
        root.addWidget(worker_group, 1)

        ppe_group = QGroupBox("Required PPE")
        ppe_grid = QGridLayout(ppe_group)
        self._ppe: dict[str, QCheckBox] = {}
        for index, class_name in enumerate(self._ppe_classes):
            checkbox = QCheckBox(class_name)
            checkbox.setObjectName(f"ppe_{class_name}")
            self._ppe[class_name] = checkbox
            ppe_grid.addWidget(checkbox, index // 3, index % 3)
        if not self._ppe:
            ppe_grid.addWidget(QLabel("No deployable PPE classes are configured."), 0, 0)
        root.addWidget(ppe_group)

        self._error = QLabel()
        self._error.setObjectName("error")
        self._error.setWordWrap(True)
        self._error.hide()
        root.addWidget(self._error)
        self._skipped_notice = QLabel()
        self._skipped_notice.setObjectName("skippedWorkerNotice")
        self._skipped_notice.setWordWrap(True)
        self._skipped_notice.hide()
        root.addWidget(self._skipped_notice)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self._cancel = QPushButton("Cancel")
        self._cancel.setObjectName("cancelWorksiteFormButton")
        self._save = QPushButton("Save" if self._editing else "Create")
        self._save.setObjectName("saveWorksiteButton")
        actions.addWidget(self._cancel)
        actions.addWidget(self._save)
        root.addLayout(actions)

        self._add_worker.clicked.connect(self.worker_profile_requested)
        self._remove_worker.clicked.connect(self.remove_selected_worker)
        self._cancel.clicked.connect(self.cancel_requested)
        self._save.clicked.connect(self.submit)

        if draft is not None:
            self.set_draft(draft)

    @property
    def is_editing(self) -> bool:
        return self._editing

    @property
    def source(self) -> str | None:
        return self._source

    @property
    def active(self) -> bool:
        return self._active

    @property
    def workers(self) -> tuple[WorksiteWorker, ...]:
        return tuple(self._workers)

    def set_draft(self, draft: WorksiteDraft) -> None:
        self._editing = True
        self._title.setText("Edit Wo.No.")
        self._save.setText("Save")
        self._id.setText(draft.worksite_id)
        self._id.setReadOnly(True)
        self._name.setText(draft.name)
        self._source = draft.source
        self._active = draft.active
        self._workers = list(draft.authorized_workers)
        self._refresh_worker_table()
        if draft.invalid_worker_messages:
            self._skipped_notice.setText(
                f"{len(draft.invalid_worker_messages)} invalid worker record(s) are excluded. "
                "Saving will explicitly omit those invalid records. Review Details first if needed."
            )
            self._skipped_notice.show()
        else:
            self._skipped_notice.hide()
        selected_ppe = set(draft.required_ppe)
        for class_name, checkbox in self._ppe.items():
            checkbox.setChecked(class_name in selected_ppe)
        unknown = sorted(selected_ppe.difference(self._ppe))
        self._unknown_ppe = tuple(unknown)
        if unknown:
            self.show_error(
                "The existing file contains unavailable PPE classes: " + ", ".join(unknown)
            )

    def add_worker_profile(self, profile: WorkerProfile) -> bool:
        return self.add_worker(
            WorksiteWorker(
                profile.worker_id,
                profile.name,
                face_image=profile.face_image,
            )
        )

    def add_worker(self, worker: WorksiteWorker) -> bool:
        error = _worker_validation_error(worker)
        if error:
            self.show_error(error)
            return False
        if any(existing.worker_id == worker.worker_id for existing in self._workers):
            self.show_error(f"Worker ID {worker.worker_id} is already in this Wo.No.")
            return False
        self._workers.append(worker)
        self._refresh_worker_table()
        self.show_error("")
        return True

    def remove_selected_worker(self) -> bool:
        row = self._worker_table.currentRow()
        if not 0 <= row < len(self._workers):
            self.show_error("Select a worker to remove.")
            return False
        del self._workers[row]
        self._refresh_worker_table()
        self.show_error("")
        return True

    def validation_error(self) -> str | None:
        if not self._id.text().strip():
            return "Worksite ID is required."
        if not self._name.text().strip():
            return "Worksite name is required."
        if not self._workers:
            return "At least one valid Worker Profile is required."
        if self._unknown_ppe:
            return "Unavailable PPE classes must be resolved: " + ", ".join(
                self._unknown_ppe
            )
        seen: set[str] = set()
        for worker in self._workers:
            error = _worker_validation_error(worker)
            if error:
                return error
            if worker.worker_id in seen:
                return f"Duplicate Worker ID: {worker.worker_id}"
            seen.add(worker.worker_id)
        return None

    def payload(self) -> dict[str, object]:
        error = self.validation_error()
        if error:
            raise ValueError(error)
        return {
            "worksite_id": self._id.text().strip(),
            "name": self._name.text().strip(),
            "authorized_workers": [
                _worksite_worker_payload(worker)
                for worker in self._workers
            ],
            "required_ppe": [
                class_name
                for class_name, checkbox in self._ppe.items()
                if checkbox.isChecked()
            ],
        }

    def submit(self) -> bool:
        try:
            payload = self.payload()
        except ValueError as error:
            self.show_error(str(error))
            return False
        self.show_error("")
        self.save_requested.emit(payload)
        return True

    def show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.setVisible(bool(message))

    def _refresh_worker_table(self) -> None:
        self._worker_table.setRowCount(len(self._workers))
        for row, worker in enumerate(self._workers):
            self._worker_table.setItem(row, 0, QTableWidgetItem(worker.worker_id))
            self._worker_table.setItem(row, 1, QTableWidgetItem(worker.name))


class WorksiteDetailsDialog(QDialog):
    """Read-only validation and worker details for a managed file."""

    def __init__(
        self,
        entry: ManagedWorksite | None = None,
        draft: WorksiteDraft | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wo.No. Details")
        self.resize(620, 480)
        layout = QVBoxLayout(self)
        self._text = QPlainTextEdit()
        self._text.setObjectName("worksiteDetailsText")
        self._text.setReadOnly(True)
        layout.addWidget(self._text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if draft is not None:
            self.set_draft(draft)
            if entry is not None and entry.issues:
                current = self._text.toPlainText()
                issue_lines = "\n".join(_format_issue(issue) for issue in entry.issues)
                self._text.setPlainText(f"{current}\n\nValidation issues and warnings:\n{issue_lines}")
        elif entry is not None:
            self.set_entry(entry)

    def set_entry(self, entry: ManagedWorksite) -> None:
        lines = [
            f"Worksite ID: {entry.worksite_id or 'Unavailable'}",
            f"Worksite name: {entry.name or '-'}",
            f"Source: {entry.source}",
            f"State: {'Available' if entry.available else 'Invalid / unavailable'}",
            f"Authorized worker count: {entry.valid_worker_count}",
            f"Skipped invalid workers: {entry.invalid_worker_count}",
            f"Required PPE: {', '.join(entry.required_ppe) or 'None'}",
        ]
        if entry.unavailable_reason:
            lines.extend(("", "Unavailable reason:", entry.unavailable_reason))
        if entry.issues:
            lines.extend(("", "Validation issues:"))
            lines.extend(_format_issue(issue) for issue in entry.issues)
        self._text.setPlainText("\n".join(lines))

    def set_draft(self, draft: WorksiteDraft) -> None:
        lines = [
            f"Worksite ID: {draft.worksite_id}",
            f"Worksite name: {draft.name}",
            f"Source: {draft.source or '-'}",
            f"Authorized worker count: {len(draft.authorized_workers)}",
            f"Required PPE: {', '.join(draft.required_ppe) or 'None'}",
            "",
            "Workers:",
        ]
        lines.extend(
            f"• {worker.worker_id} — {worker.name}" for worker in draft.authorized_workers
        )
        if draft.invalid_worker_messages:
            lines.extend(("", "Skipped invalid workers:"))
            lines.extend(f"• {message}" for message in draft.invalid_worker_messages)
        self._text.setPlainText("\n".join(lines))


class JsonViewerDialog(QDialog):
    """Read-only pretty/raw JSON viewer."""

    def __init__(self, document: RawJsonDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"View JSON — {Path(document.source).name}")
        self.resize(720, 560)
        layout = QVBoxLayout(self)
        source = QLabel(document.source)
        source.setObjectName("jsonSource")
        source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(source)
        self._parse_error = QLabel()
        self._parse_error.setObjectName("error")
        self._parse_error.setWordWrap(True)
        if document.parse_error:
            self._parse_error.setText(f"JSON parsing error: {document.parse_error}")
        else:
            self._parse_error.hide()
        layout.addWidget(self._parse_error)
        self._text = QPlainTextEdit()
        self._text.setObjectName("rawJsonText")
        self._text.setReadOnly(True)
        display = document.raw if document.parse_error else document.formatted
        self._text.setPlainText(display)
        layout.addWidget(self._text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ImportSummaryDialog(QDialog):
    """Display one non-blocking-friendly summary for a whole import batch."""

    def __init__(self, summary: ImportSummary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wo.No. Import")
        self.resize(560, 420)
        layout = QVBoxLayout(self)
        title = QLabel("Import complete")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        self._text = QPlainTextEdit()
        self._text.setObjectName("worksiteImportSummary")
        self._text.setReadOnly(True)
        self._text.setPlainText(_format_import_summary(summary))
        layout.addWidget(self._text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class UrlImportDialog(QDialog):
    """Collect exactly one direct HTTP(S) JSON URL."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Wo.No. from URL")
        self.resize(560, 150)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Direct JSON URL (http:// or https://):"))
        self._url = QLineEdit()
        self._url.setObjectName("worksiteImportUrl")
        self._url.setPlaceholderText("https://example.invalid/worksite.json")
        layout.addWidget(self._url)
        self._error = QLabel()
        self._error.setObjectName("error")
        self._error.setWordWrap(True)
        self._error.hide()
        layout.addWidget(self._error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(self._validate_then_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def url(self) -> str:
        return self._url.text().strip()

    @staticmethod
    def validation_error(url: str) -> str | None:
        parsed = urlsplit(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Enter one direct http:// or https:// JSON URL."
        if parsed.username or parsed.password:
            return "Authenticated URLs are not supported."
        return None

    def _validate_then_accept(self) -> None:
        error = self.validation_error(self.url())
        if error:
            self._error.setText(error)
            self._error.show()
            return
        self._error.hide()
        self.accept()


class WorkerProfileEditor(QDialog):
    """Minimal Worker Profile creator/editor for identity and a reusable face image."""

    save_requested = Signal(object)
    cancel_requested = Signal()

    def __init__(
        self,
        profile: WorkerProfile | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._editing = profile is not None
        self._face_image = ""
        self._source = None if profile is None else profile.source
        self.setWindowTitle("Edit Worker Profile" if self._editing else "Create Worker Profile")
        self.resize(650, 260)
        layout = QVBoxLayout(self)

        fields = QFormLayout()
        self._id = QLineEdit()
        self._id.setObjectName("workerProfileId")
        self._name = QLineEdit()
        self._name.setObjectName("workerProfileName")
        fields.addRow("Worker ID:", self._id)
        fields.addRow("Name:", self._name)
        layout.addLayout(fields)

        image_row = QHBoxLayout()
        self._image_status = QLabel("No face image loaded")
        self._image_status.setObjectName("workerProfileImageStatus")
        self._image_button = QPushButton("Choose Face Image")
        self._image_button.setObjectName("chooseWorkerFaceImageButton")
        image_row.addWidget(self._image_status, 1)
        image_row.addWidget(self._image_button)
        layout.addLayout(image_row)
        guidance = QLabel(
            "Choose a clear face image. The image is stored in the Worker Profile and "
            "can be reused when adding this worker to a Wo.No. The AI Service generates "
            "the SFace embedding only when that Wo.No. is created or saved."
        )
        guidance.setObjectName("subtleText")
        guidance.setWordWrap(True)
        layout.addWidget(guidance)

        self._error = QLabel()
        self._error.setObjectName("error")
        self._error.setWordWrap(True)
        self._error.hide()
        layout.addWidget(self._error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.submit)
        buttons.rejected.connect(self._cancelled)
        layout.addWidget(buttons)
        self._image_button.clicked.connect(self._choose_face_image)

        if profile is not None:
            self._id.setText(profile.worker_id)
            self._id.setReadOnly(True)
            self._name.setText(profile.name)
            self.set_face_image_data_uri(profile.face_image)

    @property
    def is_editing(self) -> bool:
        return self._editing

    def set_face_image_data_uri(self, value: str) -> None:
        self._face_image = value
        if value.startswith("data:image/"):
            mime = value[5:].partition(";")[0]
            self._image_status.setText(f"Loaded {mime} image")
        else:
            self._image_status.setText("Invalid face image data")

    def load_face_image(self, path: str | Path) -> None:
        source = Path(path)
        mime, _encoding = mimetypes.guess_type(source.name)
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("Choose a JPEG, PNG, or WebP image file.")
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        self.set_face_image_data_uri(f"data:{mime};base64,{encoded}")

    def profile(self) -> WorkerProfile:
        try:
            return parse_worker_profile(
                {
                    "worker_id": self._id.text().strip(),
                    "name": self._name.text().strip(),
                    "face_image": self._face_image,
                    "source": self._source,
                }
            )
        except GUIProtocolError as error:
            raise ValueError(str(error)) from error

    def payload(self) -> dict[str, object]:
        profile = self.profile()
        return {
            "worker_id": profile.worker_id,
            "name": profile.name,
            "face_image": profile.face_image,
        }

    def submit(self) -> bool:
        try:
            profile = self.profile()
        except ValueError as error:
            self._error.setText(str(error))
            self._error.show()
            return False
        self._error.hide()
        self.save_requested.emit(profile)
        return True

    def _choose_face_image(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose face image",
            "",
            "Images (*.jpg *.jpeg *.png *.webp);;All files (*)",
        )
        if not path:
            return
        try:
            self.load_face_image(path)
        except (OSError, ValueError) as error:
            self._error.setText(str(error))
            self._error.show()

    def _cancelled(self) -> None:
        self.cancel_requested.emit()
        self.reject()


def _worker_validation_error(worker: WorksiteWorker) -> str | None:
    try:
        parse_worksite_worker(_worksite_worker_payload(worker))
    except GUIProtocolError as error:
        return f"Worker {worker.worker_id}: {error}"
    return None


def _worksite_worker_payload(worker: WorksiteWorker) -> dict[str, object]:
    payload: dict[str, object] = {
        "worker_id": worker.worker_id,
        "name": worker.name,
    }
    if worker.embedding is not None:
        payload["embedding"] = list(worker.embedding)
    elif worker.face_image is not None:
        payload["face_image"] = worker.face_image
    return payload


def _format_issue(issue: object) -> str:
    severity = getattr(issue, "severity", "DATA ERROR")
    message = getattr(issue, "message", str(issue))
    worker_id = getattr(issue, "worker_id", None)
    context = f"Worker {worker_id}: " if worker_id else ""
    return f"• {severity}: {context}{message}"


def _format_import_summary(summary: ImportSummary) -> str:
    sections = (
        ("Imported", summary.imported),
        ("Skipped", summary.skipped),
        ("Failed", summary.failed),
        ("Worker warnings", summary.worker_warnings),
    )
    lines: list[str] = []
    for heading, values in sections:
        lines.append(f"{heading}:")
        lines.extend(f"• {value}" for value in values)
        if not values:
            lines.append("None")
        lines.append("")
    return "\n".join(lines).rstrip()
