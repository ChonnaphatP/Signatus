from __future__ import annotations

import io
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from signatus_launcher.supervisor import ManagedProcess, RuntimeLog


class _FakePopen:
    def __init__(self, command: list[str], **kwargs: object) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = 43210
        self.stdout = io.StringIO("child output\n")
        self.wait_calls = 0

    def poll(self) -> None:
        return None

    def wait(self, timeout: float) -> int:
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return -signal.SIGKILL


class ManagedProcessTests(unittest.TestCase):
    def test_file_log_contains_launch_process_exit_and_validation_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_log = RuntimeLog(
                Path(directory) / "logs",
                console=io.StringIO(),
                launch_id="launch-123",
            )
            path = runtime_log.path
            runtime_log.warning(
                "Core",
                "record disabled",
                pid=2468,
                exit_code=7,
                validation_severity="DATA_ERROR",
            )
            runtime_log.close()
            contents = path.read_text(encoding="utf-8")

        self.assertIn("launch=launch-123", contents)
        self.assertIn("[Core]", contents)
        self.assertIn("pid=2468", contents)
        self.assertIn("exit=7", contents)
        self.assertIn("severity=DATA_ERROR", contents)

    def test_uses_isolated_group_and_escalates_sigint_sigterm_sigkill(self) -> None:
        created: list[_FakePopen] = []

        def create(command: list[str], **kwargs: object) -> _FakePopen:
            process = _FakePopen(command, **kwargs)
            created.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            log = RuntimeLog(Path(directory) / "logs", console=output)
            process = ManagedProcess(
                "AI",
                ("python", "-m", "service"),
                cwd=Path(directory),
                environment={"SAFE": "1"},
                runtime_log=log,
                popen=create,  # type: ignore[arg-type]
            )
            try:
                with patch("signatus_launcher.supervisor.os.killpg") as kill_group:
                    process.start()
                    process.stop(0.01)
            finally:
                log.close()

        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].kwargs["start_new_session"])
        self.assertEqual(created[0].kwargs["cwd"], Path(directory))
        self.assertEqual(created[0].kwargs["env"], {"SAFE": "1"})
        self.assertIn("[AI] child output", output.getvalue())
        delivered_signals = [
            item.args[1] for item in kill_group.call_args_list if item.args[1] != 0
        ]
        self.assertEqual(
            delivered_signals,
            [
                signal.SIGINT,
                signal.SIGTERM,
                signal.SIGKILL,
            ],
        )

    def test_stops_after_graceful_stage_without_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = RuntimeLog(Path(directory) / "logs", console=io.StringIO())
            process = ManagedProcess(
                "Core",
                ("python", "-m", "service"),
                cwd=Path(directory),
                environment={},
                runtime_log=log,
                popen=_FakePopen,  # type: ignore[arg-type]
            )
            try:
                with (
                    patch("signatus_launcher.supervisor.os.killpg") as kill_group,
                    patch.object(process, "_group_exists", return_value=True),
                    patch.object(process, "_wait_for_group_exit", return_value=True),
                ):
                    process.start()
                    stopped = process.stop(0.01)
            finally:
                log.close()

        self.assertTrue(stopped)
        self.assertEqual(kill_group.call_args_list, [call(43210, signal.SIGINT)])

    def test_signals_process_group_even_when_group_leader_already_exited(self) -> None:
        class ExitedPopen(_FakePopen):
            def poll(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as directory:
            log = RuntimeLog(Path(directory) / "logs", console=io.StringIO())
            process = ManagedProcess(
                "AI",
                ("python", "-m", "service"),
                cwd=Path(directory),
                environment={},
                runtime_log=log,
                popen=ExitedPopen,  # type: ignore[arg-type]
            )
            try:
                with (
                    patch("signatus_launcher.supervisor.os.killpg") as kill_group,
                    patch.object(process, "_group_exists", return_value=True),
                    patch.object(process, "_wait_for_group_exit", return_value=True),
                ):
                    process.start()
                    stopped = process.stop(0.01)
            finally:
                log.close()

        self.assertTrue(stopped)
        kill_group.assert_called_once_with(43210, signal.SIGINT)


if __name__ == "__main__":
    unittest.main()
