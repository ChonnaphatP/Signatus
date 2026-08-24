from __future__ import annotations

import os
import unittest

try:
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication, QLabel

    from signatus_contracts.frame_buffer import FrameDetection
    from signatus_gui.client import CoreClient
    from signatus_gui.contracts import (
        CameraState,
        CameraStatus,
        Outcome,
        OutcomeStatus,
        ValidationIssue,
        ValidationReport,
        ValidationSeverity,
        Worksite,
    )
    from signatus_gui.views import DetectionScreen, MainWindow, WorksiteSelection

    PYSIDE_AVAILABLE = True
except ModuleNotFoundError:
    PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 GUI extra is not installed")
class DetectionScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.application = QApplication.instance() or QApplication([])

    def test_one_person_warning_is_persistent(self) -> None:
        screen = DetectionScreen()

        warning = screen.findChild(QLabel, "onePersonWarning")

        self.assertIsNotNone(warning)
        self.assertEqual(warning.text(), "ONE PERSON AT A TIME")
        self.assertFalse(warning.isHidden())

    def test_unavailable_preview_does_not_block_core_outcome(self) -> None:
        screen = DetectionScreen()
        screen.set_camera_available(False)

        screen.render_outcome(Outcome(status=OutcomeStatus.AUTHORIZED, worker_id="EMP0017"))

        camera_status = screen.findChild(QLabel, "cameraUnavailable")
        outcome_title = screen.findChild(QLabel, "outcomeTitle")
        self.assertIsNotNone(camera_status)
        self.assertIn("Camera stopped", camera_status.text())
        self.assertEqual(outcome_title.text(), "Access authorized")

    def test_camera_frame_replaces_unavailable_message(self) -> None:
        screen = DetectionScreen()
        frame = QImage(4, 2, QImage.Format.Format_BGR888)
        frame.fill(0xFF336699)

        screen.render_frame(frame)

        camera_status = screen.findChild(QLabel, "cameraUnavailable")
        self.assertTrue(camera_status.isHidden())
        self.assertFalse(screen.camera._image.isNull())

    def test_detection_mapping_accounts_for_letterboxing(self) -> None:
        screen = DetectionScreen()
        screen.camera.resize(800, 600)
        frame = QImage(400, 200, QImage.Format.Format_BGR888)
        detection = FrameDetection("person", 0.97, 100, 50, 300, 150)

        screen.camera.show_frame(frame, (detection,))
        mapped = screen.camera.map_detection_rect(detection)

        self.assertAlmostEqual(mapped.left(), 200, delta=1)
        self.assertAlmostEqual(mapped.top(), 200, delta=1)
        self.assertAlmostEqual(mapped.width(), 400, delta=1)
        self.assertAlmostEqual(mapped.height(), 200, delta=1)

    def test_worksite_selection_requires_explicit_select(self) -> None:
        selection = WorksiteSelection()
        worksite = Worksite("WO-014", "North Gate", ("helmet",))
        chosen: list[Worksite] = []
        selection.worksite_chosen.connect(chosen.append)
        selection.set_worksites((worksite,))

        selection._list.setCurrentRow(0)

        self.assertEqual(chosen, [])
        self.assertEqual(selection._selected_number.text(), "WO-014")
        selection._select_button.click()
        self.assertEqual(chosen, [worksite])

    def test_unavailable_worksite_is_visible_but_cannot_be_selected(self) -> None:
        selection = WorksiteSelection()
        unavailable = Worksite(
            "WO-004",
            "Unsafe policy",
            available=False,
            unavailable_reason="Missing PPE configuration",
        )
        chosen: list[Worksite] = []
        selection.worksite_chosen.connect(chosen.append)
        selection.set_worksites((unavailable,))

        selection._list.setCurrentRow(0)
        selection._select_button.click()

        self.assertEqual(chosen, [])
        self.assertFalse(selection._select_button.isEnabled())
        self.assertIn("Missing PPE", selection._error.text())
        self.assertIn("Unavailable", selection._list.item(0).text())

    def test_invalid_workers_do_not_disable_an_otherwise_valid_worksite(self) -> None:
        selection = WorksiteSelection()
        partial = Worksite(
            "WO-002",
            "North Gate",
            invalid_worker_count=2,
            valid_worker_count=48,
        )
        chosen: list[Worksite] = []
        selection.worksite_chosen.connect(chosen.append)
        selection.set_worksites((partial,))

        selection._list.setCurrentRow(0)
        selection._select_button.click()

        self.assertEqual(chosen, [partial])
        self.assertIn("2 worker", selection._error.text())

    def test_camera_controls_follow_explicit_camera_state(self) -> None:
        screen = DetectionScreen()
        starts: list[bool] = []
        stops: list[bool] = []
        screen.start_camera_requested.connect(lambda: starts.append(True))
        screen.stop_camera_requested.connect(lambda: stops.append(True))

        screen.set_camera_status(CameraStatus(CameraState.STOPPED))
        screen._start_camera.click()
        screen.set_camera_status(CameraStatus(CameraState.RUNNING))
        screen._stop_camera.click()
        screen.set_camera_status(CameraStatus(CameraState.ERROR, "camera busy"))

        self.assertEqual(starts, [True])
        self.assertEqual(stops, [True])
        self.assertIn("camera busy", screen._camera_state.text())
        self.assertTrue(screen.camera._image.isNull())

    def test_return_button_is_available_only_in_standby(self) -> None:
        screen = DetectionScreen()
        returns: list[bool] = []
        screen.return_to_selection_requested.connect(lambda: returns.append(True))

        self.assertEqual(
            screen._return_to_selection.objectName(),
            "returnToWorksiteSelectionButton",
        )
        self.assertTrue(screen._return_to_selection.isEnabled())
        screen.set_core_state("AUTHORIZATION")
        self.assertFalse(screen._return_to_selection.isEnabled())
        screen._return_to_selection.click()
        self.assertEqual(returns, [])

        screen.set_core_state("STANDBY")
        screen._return_to_selection.click()

        self.assertEqual(returns, [True])

    def test_stopped_camera_returns_to_selector_immediately(self) -> None:
        window = MainWindow()
        stops: list[bool] = []
        opened: list[bool] = []
        window.camera_stop_for_selection_requested.connect(lambda: stops.append(True))
        window.worksite_selection_opened.connect(lambda: opened.append(True))
        window.show_detection(Worksite("WO-014", "North Gate", ("helmet",)))
        window.set_camera_status(CameraStatus(CameraState.STOPPED))

        window.detection._return_to_selection.click()

        self.assertIs(window.pages.currentWidget(), window.selection)
        self.assertEqual(stops, [])
        self.assertEqual(opened, [True])
        self.assertEqual(window.detection._worksite_number.text(), "WO-014")

    def test_running_camera_is_stopped_before_returning_to_selector(self) -> None:
        window = MainWindow()
        stops: list[bool] = []
        window.camera_stop_for_selection_requested.connect(lambda: stops.append(True))
        window.show_detection(Worksite("WO-014", "North Gate", ("helmet",)))
        window.set_camera_status(CameraStatus(CameraState.RUNNING))

        window.detection._return_to_selection.click()

        self.assertEqual(stops, [True])
        self.assertIs(window.pages.currentWidget(), window.detection)
        self.assertFalse(window.detection._return_to_selection.isEnabled())
        self.assertIn("Stopping Camera", window.detection._return_to_selection.text())

        window.set_camera_status(CameraStatus(CameraState.STOPPING))
        self.assertIs(window.pages.currentWidget(), window.detection)
        window.set_camera_status(CameraStatus(CameraState.STOPPED))

        self.assertIs(window.pages.currentWidget(), window.selection)

    def test_camera_stop_failure_keeps_screening_visible(self) -> None:
        window = MainWindow()
        window.show_detection(Worksite("WO-014", "North Gate", ("helmet",)))
        window.set_camera_status(CameraStatus(CameraState.RUNNING))
        window.detection._return_to_selection.click()

        error = CameraStatus(CameraState.ERROR, "Camera could not be stopped")
        window.set_camera_status(error)
        self.assertFalse(window.detection._return_to_selection.isEnabled())
        window.selection_camera_stop_completed(error)

        self.assertIs(window.pages.currentWidget(), window.detection)
        self.assertTrue(window.detection._return_to_selection.isEnabled())
        self.assertIn("Unable to return", window.statusBar().currentMessage())

    def test_camera_stop_response_has_dedicated_completion_and_failure_signals(self) -> None:
        client = CoreClient("http://127.0.0.1:8000")
        completed: list[CameraStatus] = []
        failures: list[str] = []
        client.camera_stop_completed.connect(completed.append)
        client.camera_stop_failed.connect(failures.append)

        client._handle_stop_camera_status({"state": "STOPPED", "error": None})
        client._handle_stop_camera_status({"state": "UNKNOWN", "error": None})
        client.close()

        self.assertEqual(completed, [CameraStatus(CameraState.STOPPED)])
        self.assertEqual(len(failures), 1)
        self.assertIn("unknown camera state", failures[0])

    def test_validation_errors_create_one_summarized_dialog(self) -> None:
        window = MainWindow()
        report = ValidationReport(
            (
                ValidationIssue(
                    ValidationSeverity.DATA_ERROR,
                    "SFACE_DESCRIPTOR_LENGTH",
                    "Expected 128 values, found 3.",
                    worksite_id="002",
                    worker_id="W017",
                ),
                ValidationIssue(
                    ValidationSeverity.DATA_ERROR,
                    "WORKER_RECORD_INVALID",
                    "Worker record is invalid.",
                    worksite_id="002",
                    worker_id="W018",
                ),
            )
        )

        window.show_validation_report(report)
        first = window._validation_dialog
        window.show_validation_report(report)

        self.assertIsNotNone(first)
        self.assertIs(window._validation_dialog, first)
        self.assertIn("2 invalid worker records", first.informativeText())
        first.close()

    def test_ppe_is_neutral_until_core_outcome(self) -> None:
        screen = DetectionScreen()
        screen.set_worksite(Worksite("WO-014", "North Gate", ("helmet", "gloves")))

        self.assertEqual(screen._ppe_states["helmet"].text(), "-")
        self.assertEqual(screen._ppe_states["gloves"].text(), "-")

        screen.render_outcome(
            Outcome(
                status=OutcomeStatus.PPE_VIOLATION,
                worker_id="EMP0017",
                missing_ppe=("gloves",),
            )
        )
        self.assertIn("Pass", screen._ppe_states["helmet"].text())
        self.assertIn("Fail", screen._ppe_states["gloves"].text())


if __name__ == "__main__":
    unittest.main()
