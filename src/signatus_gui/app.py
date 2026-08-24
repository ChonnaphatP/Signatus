from __future__ import annotations

import argparse
import os
import sys

from PySide6.QtWidgets import QApplication

from .client import CoreClient
from .contracts import CameraState
from .frame_reader import SharedMemoryPreview
from .maintenance import MaintenanceCoordinator
from .views import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Signatus HDMI display client")
    parser.add_argument(
        "--core-url",
        default=os.getenv("SIGNATUS_CORE_URL", "http://127.0.0.1:8000"),
        help="Signatus Core HTTP base URL",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run in a normal window instead of full-screen kiosk mode",
    )
    parser.add_argument(
        "--frame-shm-name",
        default=os.getenv("SIGNATUS_FRAME_SHM_NAME", "signatus_camera_v1"),
        help="AI Service shared-memory preview segment name",
    )
    parser.add_argument(
        "--frame-stale-seconds",
        default=float(os.getenv("SIGNATUS_FRAME_STALE_SECONDS", "3.0")),
        type=float,
        help="Seconds without a new frame before the preview reconnects",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    application = QApplication(sys.argv[:1])
    application.setApplicationName("Signatus")

    window = MainWindow()
    client = CoreClient(args.core_url, window)
    maintenance = MaintenanceCoordinator(window, client)
    window._maintenance_coordinator = maintenance
    preview = SharedMemoryPreview(
        args.frame_shm_name,
        window,
        stale_after_seconds=args.frame_stale_seconds,
    )
    window.selection.worksite_chosen.connect(client.select_worksite)
    client.worksites_loaded.connect(window.selection.set_worksites)
    client.worksite_selected.connect(window.show_detection)
    client.worksite_selected.connect(lambda _worksite: window.selection.set_enabled(True))
    client.outcome_received.connect(window.detection.render_outcome)
    client.connection_changed.connect(window.set_connected)
    client.state_changed.connect(window.detection.set_core_state)
    client.camera_status_changed.connect(window.set_camera_status)
    client.camera_status_changed.connect(
        lambda status: preview.set_camera_running(status.state is CameraState.RUNNING)
    )
    client.validation_report_loaded.connect(window.show_validation_report)
    client.request_failed.connect(window.selection.show_error)
    client.protocol_error.connect(window.selection.show_error)
    client.request_failed.connect(lambda _message: window.selection.set_enabled(True))
    client.protocol_error.connect(lambda _message: window.selection.set_enabled(True))
    preview.frame_ready.connect(window.detection.render_frame)
    preview.availability_changed.connect(window.detection.set_camera_available)
    window.detection.start_camera_requested.connect(client.start_camera)
    window.detection.stop_camera_requested.connect(client.stop_camera)
    window.camera_stop_for_selection_requested.connect(client.stop_camera)
    window.worksite_selection_opened.connect(client.load_worksites)
    client.camera_stop_completed.connect(window.selection_camera_stop_completed)
    client.camera_stop_failed.connect(window.cancel_selection_return)
    application.aboutToQuit.connect(client.close)
    application.aboutToQuit.connect(preview.close)

    window.selection.worksite_chosen.connect(lambda _worksite: window.selection.set_enabled(False))
    client.load_worksites()
    client.load_validation_report()
    client.connect_outcomes()
    client.start_state_monitoring()
    preview.start()

    if args.windowed:
        window.show()
    else:
        window.showFullScreen()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
