from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from signatus_contracts.frame_buffer import FrameDetection

from .contracts import (
    CameraState,
    CameraStatus,
    Outcome,
    OutcomeStatus,
    ValidationIssue,
    ValidationReport,
    Worksite,
)
from .worksite_views import WorksiteForm, WorksiteManager

STYLESHEET = """
QWidget {
    background: #f0f0f0;
    color: #202020;
    font-family: "Segoe UI";
    font-size: 13px;
}
QMainWindow { background: #f0f0f0; }
QLabel#applicationName { font-size: 22px; font-weight: 600; }
QLabel#pageTitle { font-size: 16px; font-weight: 600; }
QLabel#versionText, QLabel#fieldCaption, QLabel#subtleText { color: #5b5b5b; }
QLabel#error { color: #a1260d; }
QLabel#systemStatus {
    background: #e5e5e5;
    border: 1px solid #a9a9a9;
    padding: 7px 8px;
    font-weight: 600;
}
QLabel#onePersonWarning {
    background: #fff4ce;
    border: 1px solid #c8a100;
    padding: 7px;
    font-weight: 600;
}
QLabel#outcomeTitle { font-size: 15px; font-weight: 600; }
QLabel#ppePass { color: #176b2c; font-weight: 600; }
QLabel#ppeFail { color: #a1260d; font-weight: 600; }
QLabel#ppeNeutral { color: #5b5b5b; font-weight: 600; }
QFrame#separator { color: #c8c8c8; }
QFrame#camera { background: #171717; border: 1px solid #777777; }
QLabel#cameraUnavailable { background: transparent; color: #e6e6e6; }
QListWidget { background: white; border: 1px solid #7a7a7a; outline: 0; }
QListWidget::item { min-height: 28px; padding: 2px 6px; }
QListWidget::item:selected { background: #0078d7; color: white; }
QPushButton { min-width: 84px; min-height: 24px; padding: 2px 12px; }
QStatusBar { background: #e6e6e6; border-top: 1px solid #c6c6c6; }
"""

PPE_LABELS = {
    "helmet": "Helmet",
    "vest": "Safety Vest",
    "gloves": "Gloves",
    "boots": "Safety Shoes",
    "goggles": "Safety Goggles",
}
DEFAULT_PPE = ("helmet", "vest", "gloves", "boots")


def _application_version() -> str:
    try:
        return version("signatus")
    except PackageNotFoundError:
        return "0.1.0"


class WorksiteSelection(QWidget):
    worksite_chosen = Signal(object)
    manager_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._worksites: tuple[Worksite, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 28)
        layout.setSpacing(10)
        layout.addWidget(_label(f"Signatus Ver. {_application_version()}", "versionText"))
        layout.addWidget(_label("Signatus", "applicationName"))
        layout.addSpacing(10)
        layout.addWidget(_label("Select active Wo.No.", "pageTitle"))

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._list.currentRowChanged.connect(self._show_selected)
        self._list.itemDoubleClicked.connect(lambda _item: self._choose_selected())
        layout.addWidget(self._list, 1)

        details = QGridLayout()
        details.setHorizontalSpacing(12)
        details.setVerticalSpacing(5)
        details.addWidget(_label("Wo.No.:", "fieldCaption"), 0, 0)
        self._selected_number = _label("-", "selectedWorksiteNumber")
        details.addWidget(self._selected_number, 0, 1)
        details.addWidget(_label("รายละเอียด / ชื่อ Wo.No.:", "fieldCaption"), 1, 0)
        self._selected_name = _label("-", "selectedWorksiteName")
        self._selected_name.setWordWrap(True)
        details.addWidget(self._selected_name, 1, 1)
        details.setColumnStretch(1, 1)

        action_row = QHBoxLayout()
        action_row.addLayout(details, 1)
        self._select_button = QPushButton("Select")
        self._select_button.setObjectName("selectWorksiteButton")
        self._select_button.setEnabled(False)
        self._select_button.clicked.connect(self._choose_selected)
        action_row.addWidget(self._select_button, 0, Qt.AlignmentFlag.AlignBottom)
        self._manager_button = QPushButton("Manage Wo.No.")
        self._manager_button.setObjectName("manageWorksitesButton")
        self._manager_button.clicked.connect(self.manager_requested)
        action_row.addWidget(self._manager_button, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(action_row)

        self._error = _label("", "error")
        self._error.setWordWrap(True)
        self._error.hide()
        layout.addWidget(self._error)

    def set_worksites(self, worksites: tuple[Worksite, ...]) -> None:
        self._worksites = worksites
        self._list.clear()
        self._selected_number.setText("-")
        self._selected_name.setText("-")
        self._select_button.setEnabled(False)
        if not worksites:
            self.show_error("No Wo.No. configuration files were found")
            return
        self._error.hide()
        for index, worksite in enumerate(worksites, start=1):
            suffix = " — Unavailable" if not worksite.available else ""
            item = QListWidgetItem(f"{index}. {worksite.worksite_id}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, worksite)
            tooltip = worksite.name
            if not worksite.available:
                tooltip = f"{tooltip}\n{worksite.unavailable_reason or 'Configuration unavailable'}"
                item.setForeground(QBrush(QColor("#8a3b2b")))
            elif worksite.invalid_worker_count:
                tooltip = (
                    f"{tooltip}\n{worksite.invalid_worker_count} worker record(s) unavailable"
                )
            item.setToolTip(tooltip)
            self._list.addItem(item)

    def set_enabled(self, enabled: bool) -> None:
        self._list.setEnabled(enabled)
        row = self._list.currentRow()
        self._select_button.setEnabled(
            enabled and 0 <= row < len(self._worksites) and self._worksites[row].available
        )

    def show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.show()

    def _show_selected(self, row: int) -> None:
        if not 0 <= row < len(self._worksites):
            self._selected_number.setText("-")
            self._selected_name.setText("-")
            self._select_button.setEnabled(False)
            return
        worksite = self._worksites[row]
        self._selected_number.setText(worksite.worksite_id)
        self._selected_name.setText(worksite.name)
        self._select_button.setEnabled(self._list.isEnabled() and worksite.available)
        if not worksite.available:
            self.show_error(worksite.unavailable_reason or "This Wo.No. is unavailable")
        elif worksite.invalid_worker_count:
            self.show_error(
                f"{worksite.invalid_worker_count} worker record(s) are unavailable. "
                "Other valid workers remain active."
            )
        else:
            self._error.hide()

    def _choose_selected(self) -> None:
        row = self._list.currentRow()
        if self._list.isEnabled() and 0 <= row < len(self._worksites):
            worksite = self._worksites[row]
            if worksite.available:
                self.worksite_chosen.emit(worksite)


class CameraView(QFrame):
    """Paint a frame and its same-frame YOLO boxes without child box widgets."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("camera")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 320)
        self._image = QImage()
        self._detections: tuple[FrameDetection, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        self._unavailable = _label("Camera unavailable", "cameraUnavailable")
        self._unavailable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._unavailable)

    @property
    def detections(self) -> tuple[FrameDetection, ...]:
        return self._detections

    def show_frame(
        self,
        frame: QImage,
        detections: tuple[FrameDetection, ...] = (),
    ) -> None:
        if frame.isNull():
            self.show_unavailable()
            return
        self._image = frame
        self._detections = detections
        self._unavailable.hide()
        self.update()

    def show_unavailable(self, message: str = "Camera unavailable") -> None:
        self._image = QImage()
        self._detections = ()
        self._unavailable.setText(message)
        self._unavailable.show()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._image.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = self._display_rect()
        painter.drawImage(target, self._image)
        painter.setClipRect(target)
        box_pen = QPen(QColor("#ffd33d"))
        box_pen.setWidth(2)
        font = QFont(self.font())
        font.setPixelSize(12)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for detection in self._detections:
            box = self.map_detection_rect(detection)
            if box.isEmpty():
                continue
            painter.setPen(box_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)
            text = f"{detection.class_name} {detection.confidence:.2f}"
            text_width = metrics.horizontalAdvance(text) + 8
            text_height = metrics.height() + 4
            label_left = box.left()
            label_top = max(target.top(), box.top() - text_height)
            label = QRectF(
                label_left,
                label_top,
                min(text_width, max(0.0, target.right() - label_left)),
                text_height,
            )
            painter.fillRect(label, QColor(32, 32, 32, 220))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                label.adjusted(4, 1, -2, -1),
                Qt.AlignmentFlag.AlignVCenter,
                text,
            )

    def _display_rect(self) -> QRectF:
        available = QRectF(self.contentsRect())
        if self._image.isNull() or available.isEmpty():
            return QRectF()
        scale = min(
            available.width() / self._image.width(),
            available.height() / self._image.height(),
        )
        width = self._image.width() * scale
        height = self._image.height() * scale
        return QRectF(
            available.left() + (available.width() - width) / 2,
            available.top() + (available.height() - height) / 2,
            width,
            height,
        )

    def map_detection_rect(self, detection: FrameDetection) -> QRectF:
        if self._image.isNull():
            return QRectF()
        target = self._display_rect()
        x1 = max(0.0, min(float(self._image.width()), detection.x1))
        y1 = max(0.0, min(float(self._image.height()), detection.y1))
        x2 = max(0.0, min(float(self._image.width()), detection.x2))
        y2 = max(0.0, min(float(self._image.height()), detection.y2))
        scale_x = target.width() / self._image.width()
        scale_y = target.height() / self._image.height()
        return QRectF(
            target.left() + min(x1, x2) * scale_x,
            target.top() + min(y1, y2) * scale_y,
            abs(x2 - x1) * scale_x,
            abs(y2 - y1) * scale_y,
        )


class DetectionScreen(QWidget):
    start_camera_requested = Signal()
    stop_camera_requested = Signal()
    return_to_selection_requested = Signal()
    manager_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._required_ppe: tuple[str, ...] = ()
        self._ppe_states: dict[str, QLabel] = {}
        self._camera_status = CameraStatus(CameraState.STOPPED)
        self._core_state_value = "STANDBY"
        self._return_pending = False
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.camera = CameraView()
        splitter.addWidget(self.camera)

        panel = QFrame()
        panel.setObjectName("informationPanel")
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(340)
        side = QVBoxLayout(panel)
        side.setContentsMargins(14, 4, 8, 4)
        side.setSpacing(7)
        self._status = _label("SYSTEM: STANDBY", "systemStatus")
        side.addWidget(self._status)
        self._core_state = _label("Core state: STANDBY", "subtleText")
        self._core_state.setObjectName("coreState")
        side.addWidget(self._core_state)
        self._camera_state = _label("Camera: STOPPED", "subtleText")
        self._camera_state.setObjectName("cameraState")
        side.addWidget(self._camera_state)
        camera_actions = QHBoxLayout()
        self._start_camera = QPushButton("Start Camera")
        self._start_camera.setObjectName("startCameraButton")
        self._start_camera.clicked.connect(self.start_camera_requested)
        self._stop_camera = QPushButton("Stop Camera")
        self._stop_camera.setObjectName("stopCameraButton")
        self._stop_camera.clicked.connect(self.stop_camera_requested)
        camera_actions.addWidget(self._start_camera)
        camera_actions.addWidget(self._stop_camera)
        side.addLayout(camera_actions)
        self._manager_button = QPushButton("Manage Wo.No.")
        self._manager_button.setObjectName("manageWorksitesButton")
        self._manager_button.clicked.connect(self.manager_requested)
        side.addWidget(self._manager_button)
        self._return_to_selection = QPushButton("Return to Wo.No. Selection")
        self._return_to_selection.setObjectName("returnToWorksiteSelectionButton")
        self._return_to_selection.clicked.connect(self.return_to_selection_requested)
        side.addWidget(self._return_to_selection)
        side.addSpacing(4)
        side.addWidget(_label("Wo.No.", "fieldCaption"))
        self._worksite_number = _label("-", "activeWorksiteNumber")
        side.addWidget(self._worksite_number)
        side.addWidget(_label("Wo.No. Name", "fieldCaption"))
        self._worksite_name = _label("-", "activeWorksiteName")
        self._worksite_name.setWordWrap(True)
        side.addWidget(self._worksite_name)
        side.addWidget(_separator())
        side.addWidget(_label("Worker", "fieldCaption"))
        self._worker = _label("Unknown", "workerName")
        self._worker.setWordWrap(True)
        side.addWidget(self._worker)
        side.addWidget(_separator())
        side.addWidget(_label("PPE Status", "pageTitle"))
        self._ppe_grid = QGridLayout()
        self._ppe_grid.setHorizontalSpacing(12)
        self._ppe_grid.setVerticalSpacing(5)
        self._ppe_grid.setColumnStretch(0, 1)
        side.addLayout(self._ppe_grid)
        side.addWidget(_separator())
        self._outcome_title = _label("Waiting for a worker", "outcomeTitle")
        side.addWidget(self._outcome_title)
        self._detail = _label("Core is monitoring the entrance.", "outcomeDetail")
        self._detail.setWordWrap(True)
        side.addWidget(self._detail)
        side.addStretch(1)
        warning = _label("ONE PERSON AT A TIME", "onePersonWarning")
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side.addWidget(warning)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 300])
        root.addWidget(splitter)
        self._set_ppe_rows(DEFAULT_PPE)
        self.set_camera_status(self._camera_status)
        self._update_return_button()

    def set_worksite(self, worksite: Worksite) -> None:
        self._worksite_number.setText(worksite.worksite_id)
        self._worksite_name.setText(worksite.name)
        self._required_ppe = worksite.required_ppe
        self._set_ppe_rows(worksite.required_ppe or DEFAULT_PPE)
        self._set_all_ppe_neutral()
        self._status.setText("SYSTEM: STANDBY")
        self._worker.setText("Unknown")
        self._outcome_title.setText("Waiting for a worker")
        self._detail.setText("Core is monitoring the entrance.")

    def render_frame(self, preview: object) -> None:
        if isinstance(preview, QImage):
            self.camera.show_frame(preview)
        elif (
            isinstance(preview, tuple)
            and len(preview) == 2
            and isinstance(preview[0], QImage)
            and isinstance(preview[1], tuple)
        ):
            self.camera.show_frame(preview[0], preview[1])

    def set_camera_available(self, available: bool) -> None:
        if not available:
            message = {
                CameraState.STOPPED: "Camera stopped",
                CameraState.STARTING: "Camera starting…",
                CameraState.RUNNING: "Waiting for camera frames…",
                CameraState.STOPPING: "Camera stopping…",
                CameraState.ERROR: self._camera_status.error or "Camera error",
            }[self._camera_status.state]
            self.camera.show_unavailable(message)

    def set_camera_status(self, status: CameraStatus) -> None:
        self._camera_status = status
        state = status.state
        text = f"Camera: {state}"
        if status.error:
            text = f"{text} — {status.error}"
        self._camera_state.setText(text)
        self._camera_state.setToolTip(status.error or "")
        self._start_camera.setEnabled(state in {CameraState.STOPPED, CameraState.ERROR})
        self._stop_camera.setEnabled(
            state in {CameraState.STARTING, CameraState.RUNNING, CameraState.ERROR}
        )
        if state is not CameraState.RUNNING:
            self.set_camera_available(False)

    def set_core_state(self, state: str) -> None:
        self._core_state_value = state
        display = "AUTHORIZING" if state == "AUTHORIZATION" else state
        self._core_state.setText(f"Core state: {display}")
        self._update_return_button()

    @property
    def core_state(self) -> str:
        return self._core_state_value

    def set_return_pending(self, pending: bool) -> None:
        self._return_pending = pending
        self._update_return_button()

    def _update_return_button(self) -> None:
        self._return_to_selection.setEnabled(
            self._core_state_value == "STANDBY" and not self._return_pending
        )
        self._return_to_selection.setText(
            "Stopping Camera…" if self._return_pending else "Return to Wo.No. Selection"
        )
        if self._core_state_value != "STANDBY":
            tooltip = "Wait for the current authorization to finish."
        elif self._return_pending:
            tooltip = "Waiting for Core to confirm that the camera is stopped."
        else:
            tooltip = "Stop screening and return to the Wo.No. selector."
        self._return_to_selection.setToolTip(tooltip)

    def render_outcome(self, outcome: Outcome) -> None:
        self._set_all_ppe_neutral()
        if outcome.status is OutcomeStatus.AUTHORIZED:
            self._status.setText("SYSTEM: AUTHORIZED")
            self._outcome_title.setText("Access authorized")
            self._detail.setText("Identity and required PPE checks passed.")
            self._worker.setText(outcome.worker_id or "Unknown")
            for item in self._required_ppe:
                self._set_ppe_state(item, "pass")
        elif outcome.status is OutcomeStatus.PPE_VIOLATION:
            self._status.setText("SYSTEM: DENIED")
            self._outcome_title.setText("PPE violation")
            missing = {item.casefold() for item in outcome.missing_ppe}
            self._detail.setText(
                "Missing PPE: " + (", ".join(outcome.missing_ppe) or "Required PPE not observed")
            )
            self._worker.setText(outcome.worker_id or "Unknown")
            for item in self._required_ppe:
                self._set_ppe_state(item, "fail" if item.casefold() in missing else "pass")
        elif outcome.status is OutcomeStatus.UNAUTHORIZED:
            self._status.setText("SYSTEM: DENIED")
            self._outcome_title.setText("Unauthorized")
            self._detail.setText("Worker identity is not authorized for this Wo.No.")
            self._worker.setText("Unknown")
        else:
            self._status.setText("SYSTEM: FACE CAPTURE FAILED")
            self._outcome_title.setText("Face capture failed")
            reason = (outcome.face_failure_reason or "No usable face capture").replace("_", " ")
            attempt = f" Attempt {outcome.attempt} of 3." if outcome.attempt is not None else ""
            instruction = (
                " Leave the screening area before trying again."
                if outcome.retry_allowed is False
                else " Reposition for another capture."
            )
            self._detail.setText(f"{reason}.{attempt}{instruction}")
            self._worker.setText("Unknown")

    def _set_ppe_rows(self, items: tuple[str, ...]) -> None:
        while self._ppe_grid.count():
            layout_item = self._ppe_grid.takeAt(0)
            if layout_item.widget() is not None:
                layout_item.widget().hide()
                layout_item.widget().deleteLater()
        self._ppe_states.clear()
        for row, item in enumerate(items):
            key = item.casefold()
            name = QLabel(PPE_LABELS.get(key, item.replace("_", " ").title()))
            state = _label("-", "ppeNeutral")
            state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._ppe_grid.addWidget(name, row, 0)
            self._ppe_grid.addWidget(state, row, 1)
            self._ppe_states[key] = state

    def _set_all_ppe_neutral(self) -> None:
        for key in self._ppe_states:
            self._set_ppe_state(key, "neutral")

    def _set_ppe_state(self, item: str, state: str) -> None:
        label = self._ppe_states.get(item.casefold())
        if label is None:
            return
        text, object_name = {
            "pass": ("✓  Pass", "ppePass"),
            "fail": ("✕  Fail", "ppeFail"),
            "neutral": ("-", "ppeNeutral"),
        }[state]
        label.setText(text)
        label.setObjectName(object_name)
        label.style().unpolish(label)
        label.style().polish(label)


class MainWindow(QMainWindow):
    manager_opened = Signal()
    camera_stop_for_selection_requested = Signal()
    worksite_selection_opened = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Signatus")
        self.setMinimumSize(1024, 640)
        self.resize(1280, 760)
        self.setStyleSheet(STYLESHEET)
        self.pages = QStackedWidget()
        self.selection = WorksiteSelection()
        self.detection = DetectionScreen()
        self.manager = WorksiteManager()
        self._worksite_form: WorksiteForm | None = None
        self._manager_return_page: QWidget = self.selection
        self._camera_status = CameraStatus(CameraState.STOPPED)
        self._selection_return_pending = False
        self._selection_stop_acknowledged = False
        self.pages.addWidget(self.selection)
        self.pages.addWidget(self.detection)
        self.pages.addWidget(self.manager)
        self.setCentralWidget(self.pages)
        self._connection = _label("Core: Connecting", "connectionStatus")
        self.statusBar().addPermanentWidget(self._connection)
        self._camera = _label("Camera: STOPPED", "cameraConnectionStatus")
        self.statusBar().addPermanentWidget(self._camera)
        self.statusBar().showMessage("Ready")
        self._validation_dialog: QMessageBox | None = None
        self._validation_report_shown = False
        self.selection.manager_requested.connect(self.open_manager)
        self.detection.manager_requested.connect(self.open_manager)
        self.detection.return_to_selection_requested.connect(
            self.return_to_worksite_selection
        )
        self.manager.return_requested.connect(self.return_from_manager)

    def show_detection(self, worksite: Worksite) -> None:
        self._selection_return_pending = False
        self._selection_stop_acknowledged = False
        self.detection.set_return_pending(False)
        self.detection.set_worksite(worksite)
        self.pages.setCurrentWidget(self.detection)
        self.statusBar().showMessage(f"Active Wo.No.: {worksite.worksite_id}")

    def return_to_worksite_selection(self) -> None:
        if (
            self.pages.currentWidget() is not self.detection
            or self.detection.core_state != "STANDBY"
            or self._selection_return_pending
        ):
            return
        if self._camera_status.state is CameraState.STOPPED:
            self._complete_selection_return()
            return
        self._selection_return_pending = True
        self._selection_stop_acknowledged = (
            self._camera_status.state is CameraState.STOPPING
        )
        self.detection.set_return_pending(True)
        self.statusBar().showMessage("Stopping camera before returning to Wo.No. selection…")
        if self._camera_status.state is not CameraState.STOPPING:
            self.camera_stop_for_selection_requested.emit()

    def cancel_selection_return(self, message: str = "Camera stop failed") -> None:
        if not self._selection_return_pending:
            return
        self._selection_return_pending = False
        self._selection_stop_acknowledged = False
        self.detection.set_return_pending(False)
        self.statusBar().showMessage(
            f"Unable to return to Wo.No. selection: {message}"
        )

    def selection_camera_stop_completed(self, status: CameraStatus) -> None:
        if not self._selection_return_pending:
            return
        self._selection_stop_acknowledged = True
        if status.state is CameraState.STOPPED:
            self._complete_selection_return()
        elif status.state is CameraState.ERROR:
            self.cancel_selection_return(status.error or "Camera could not be stopped")

    def _complete_selection_return(self) -> None:
        self._selection_return_pending = False
        self._selection_stop_acknowledged = False
        self.detection.set_return_pending(False)
        self.selection.set_enabled(True)
        self.pages.setCurrentWidget(self.selection)
        self.statusBar().showMessage("Select active Wo.No.")
        self.worksite_selection_opened.emit()

    def open_manager(self) -> None:
        current = self.pages.currentWidget()
        if current in {self.selection, self.detection}:
            self._manager_return_page = current
        self.pages.setCurrentWidget(self.manager)
        self.statusBar().showMessage("Wo.No. maintenance")
        self.manager_opened.emit()

    def return_from_manager(self) -> None:
        self.pages.setCurrentWidget(self._manager_return_page)
        self.statusBar().showMessage("Ready")

    def show_worksite_form(self, form: WorksiteForm) -> None:
        if self._worksite_form is not None:
            self.pages.removeWidget(self._worksite_form)
            self._worksite_form.deleteLater()
        self._worksite_form = form
        self.pages.addWidget(form)
        self.pages.setCurrentWidget(form)

    def close_worksite_form(self) -> None:
        self.pages.setCurrentWidget(self.manager)
        if self._worksite_form is not None:
            self.pages.removeWidget(self._worksite_form)
            self._worksite_form.deleteLater()
            self._worksite_form = None

    def set_connected(self, connected: bool) -> None:
        self._connection.setText("Core: Connected" if connected else "Core: Reconnecting")

    def set_camera_status(self, status: CameraStatus) -> None:
        self._camera_status = status
        text = f"Camera: {status.state}"
        if status.error:
            text = f"{text} — {status.error}"
        self._camera.setText(text)
        self._camera.setToolTip(status.error or "")
        self.detection.set_camera_status(status)
        if self._selection_return_pending:
            if status.state is CameraState.STOPPED:
                self._complete_selection_return()
            elif (
                status.state is CameraState.ERROR
                and self._selection_stop_acknowledged
            ):
                self.cancel_selection_return(status.error or "Camera could not be stopped")

    def show_validation_report(self, report: ValidationReport) -> None:
        if self._validation_report_shown or not report.data_errors:
            return
        self._validation_report_shown = True
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Signatus Data Validation")
        dialog.setText("Some deployment data contains errors.")
        dialog.setInformativeText(_validation_summary(report))
        dialog.setDetailedText(_validation_details(report))
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.finished.connect(self._validation_dialog_closed)
        self._validation_dialog = dialog
        dialog.open()

    def _validation_dialog_closed(self, _result: int) -> None:
        self._validation_dialog = None


def _validation_summary(report: ValidationReport) -> str:
    grouped: dict[str, list[ValidationIssue]] = {}
    for issue in report.data_errors:
        label = f"Wo.No. {issue.worksite_id}" if issue.worksite_id else issue.source or "Deployment"
        grouped.setdefault(label, []).append(issue)

    lines: list[str] = []
    for label, issues in grouped.items():
        worker_issues = [issue for issue in issues if issue.worker_id]
        worksite_issues = [issue for issue in issues if not issue.worker_id]
        lines.append(label)
        if worker_issues:
            count = len(worker_issues)
            lines.append(f"{count} invalid worker record{'s' if count != 1 else ''}")
        lines.extend(issue.message for issue in worksite_issues[:2])
        lines.append("")
    lines.extend(
        [
            "Affected records have been disabled.",
            "Other valid data remains available.",
        ]
    )
    return "\n".join(lines)


def _validation_details(report: ValidationReport) -> str:
    lines: list[str] = []
    for issue in report.issues:
        context = [issue.severity, issue.code]
        if issue.worksite_id:
            context.append(f"Wo.No. {issue.worksite_id}")
        if issue.worker_id:
            context.append(f"Worker {issue.worker_id}")
        if issue.source:
            context.append(issue.source)
        lines.append(" | ".join(context))
        lines.append(issue.message)
        lines.append("")
    return "\n".join(lines).rstrip()


def _label(text: str, object_name: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName(object_name)
    return label


def _separator() -> QFrame:
    line = QFrame()
    line.setObjectName("separator")
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line
