from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from signatus_launcher.app import main
from signatus_launcher.instance import SingleInstanceError
from signatus_launcher.preflight import PreflightReport


class LauncherAppTests(unittest.TestCase):
    def test_check_only_never_constructs_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("", encoding="utf-8")
            output = io.StringIO()
            errors = io.StringIO()
            with (
                patch("signatus_launcher.app.run_preflight", return_value=PreflightReport()),
                patch("signatus_launcher.app.Supervisor") as supervisor,
            ):
                result = main(
                    ["--env-file", str(env_file), "--check-only", "--no-gui"],
                    stdout=output,
                    stderr=errors,
                )

        self.assertEqual(result, 0)
        self.assertIn("No processes were started", output.getvalue())
        self.assertEqual(errors.getvalue(), "")
        supervisor.assert_not_called()

    def test_preflight_failure_never_constructs_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("", encoding="utf-8")
            report = PreflightReport(errors=["camera unavailable"])
            with (
                patch("signatus_launcher.app.run_preflight", return_value=report),
                patch("signatus_launcher.app.Supervisor") as supervisor,
            ):
                result = main(
                    ["--env-file", str(env_file), "--no-gui"],
                    stdout=io.StringIO(),
                    stderr=io.StringIO(),
                )

        self.assertEqual(result, 1)
        supervisor.assert_not_called()

    def test_check_only_passes_with_data_errors_and_prints_separate_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("", encoding="utf-8")
            output = io.StringIO()
            errors = io.StringIO()
            report = PreflightReport(
                data_errors=["worker W017 descriptor is invalid"],
                warnings=["optional description missing"],
            )
            with patch("signatus_launcher.app.run_preflight", return_value=report):
                result = main(
                    ["--env-file", str(env_file), "--check-only", "--no-gui"],
                    stdout=output,
                    stderr=errors,
                )

        self.assertEqual(result, 0)
        self.assertIn("Data errors:", output.getvalue())
        self.assertIn("Warnings:", output.getvalue())
        self.assertIn("RESULT: PASS WITH DATA ERRORS", output.getvalue())
        self.assertEqual(errors.getvalue(), "")

    def test_single_instance_failure_never_runs_preflight_or_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("", encoding="utf-8")
            errors = io.StringIO()
            with (
                patch(
                    "signatus_launcher.app.SingleInstanceLock.acquire",
                    side_effect=SingleInstanceError("already running"),
                ),
                patch("signatus_launcher.app.run_preflight") as preflight,
                patch("signatus_launcher.app.Supervisor") as supervisor,
            ):
                result = main(
                    ["--env-file", str(env_file), "--no-gui"],
                    stdout=io.StringIO(),
                    stderr=errors,
                )

        self.assertEqual(result, 1)
        self.assertIn("already running", errors.getvalue())
        self.assertIn("RESULT: FAIL", errors.getvalue())
        preflight.assert_not_called()
        supervisor.assert_not_called()

    def test_data_errors_do_not_prevent_supervisor_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("", encoding="utf-8")
            output = io.StringIO()
            report = PreflightReport(data_errors=["one worker is unavailable"])
            with (
                patch("signatus_launcher.app.run_preflight", return_value=report),
                patch("signatus_launcher.app.Supervisor") as supervisor,
            ):
                supervisor.return_value.run.return_value = 0
                result = main(
                    ["--env-file", str(env_file), "--no-gui"],
                    stdout=output,
                    stderr=io.StringIO(),
                )

        self.assertEqual(result, 0)
        self.assertIn("RESULT: PASS WITH DATA ERRORS", output.getvalue())
        supervisor.assert_called_once()
        supervisor.return_value.run.assert_called_once_with()

    def test_rejects_windowed_without_gui(self) -> None:
        result = main(
            ["--windowed", "--no-gui"],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
