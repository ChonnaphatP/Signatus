from __future__ import annotations

import io
import tempfile
import unittest
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from signatus_launcher.config import LauncherConfig
from signatus_launcher.supervisor import (
    RuntimeLog,
    Supervisor,
    validate_ai_health,
    validate_core_health,
)


class _FakeProcess:
    instances: ClassVar[list[_FakeProcess]] = []
    events: ClassVar[list[str]] = []
    exit_codes: ClassVar[dict[str, int | None]] = {}

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        runtime_log: RuntimeLog,
    ) -> None:
        del runtime_log
        self.name = name
        self.command = tuple(command)
        self.cwd = cwd
        self.environment = dict(environment)
        self.instances.append(self)

    def start(self) -> None:
        self.events.append(f"start:{self.name}")

    def poll(self) -> int | None:
        return self.exit_codes.get(self.name)

    def stop(self, timeout: float) -> None:
        del timeout
        self.events.append(f"stop:{self.name}")

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.events = []
        cls.exit_codes = {"AI": None, "Core": None, "GUI": 0}


class _AlwaysReadyProbe:
    def check(
        self,
        url: str,
        validator: Callable[[Mapping[str, object]], str | None],
    ) -> tuple[bool, str]:
        payload = _core_health() if url.endswith("/api/health") else _ai_health()
        problem = validator(payload)
        return problem is None, problem or "ready"


class LauncherSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeProcess.reset()

    def test_starts_in_order_and_stops_in_reverse_after_clean_gui_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory), windowed=True)
            log = RuntimeLog(config.log_dir, console=io.StringIO())
            try:
                supervisor = Supervisor(
                    config,
                    log,
                    probe=_AlwaysReadyProbe(),
                    process_type=_FakeProcess,
                )
                result = supervisor.run()
            finally:
                log.close()

        self.assertEqual(result, 0)
        self.assertEqual(
            _FakeProcess.events,
            ["start:AI", "start:Core", "start:GUI", "stop:GUI", "stop:Core", "stop:AI"],
        )
        ai, core, gui = _FakeProcess.instances
        self.assertEqual(ai.cwd, config.project_dir)
        self.assertEqual(ai.environment, core.environment)
        self.assertEqual(core.environment, gui.environment)
        self.assertEqual(ai.environment["SIGNATUS_LAUNCH_ID"], log.launch_id)
        self.assertTrue(ai.environment["SIGNATUS_LAUNCHER_PID"].isdecimal())
        self.assertEqual(ai.command[1:4], ("-m", "uvicorn", "signatus_ai.app:app"))
        self.assertEqual(core.command[1:4], ("-m", "uvicorn", "signatus_core.app:app"))
        self.assertEqual(gui.command[-1], "--windowed")

    def test_partial_start_failure_stops_only_processes_that_were_started(self) -> None:
        _FakeProcess.exit_codes["AI"] = 7
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            log = RuntimeLog(config.log_dir, console=io.StringIO())
            try:
                result = Supervisor(
                    config,
                    log,
                    probe=_AlwaysReadyProbe(),
                    process_type=_FakeProcess,
                ).run()
            finally:
                log.close()

        self.assertEqual(result, 1)
        self.assertEqual(_FakeProcess.events, ["start:AI", "stop:AI"])

    def test_shutdown_continues_after_one_process_stop_fails(self) -> None:
        class StopFailureProcess(_FakeProcess):
            def stop(self, timeout: float) -> None:
                super().stop(timeout)
                if self.name == "GUI":
                    raise RuntimeError("simulated stop failure")

        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            log = RuntimeLog(config.log_dir, console=io.StringIO())
            try:
                result = Supervisor(
                    config,
                    log,
                    probe=_AlwaysReadyProbe(),
                    process_type=StopFailureProcess,
                ).run()
            finally:
                log.close()

        self.assertEqual(result, 1)
        self.assertEqual(
            _FakeProcess.events[-3:],
            ["stop:GUI", "stop:Core", "stop:AI"],
        )

    def test_runtime_readiness_loss_tears_down_entire_stack(self) -> None:
        class FailingAIProbe(_AlwaysReadyProbe):
            ai_calls = 0

            def check(self, url, validator):  # type: ignore[no-untyped-def]
                if url.endswith("/health") and not url.endswith("/api/health"):
                    self.ai_calls += 1
                    if self.ai_calls > 2:
                        return False, "AI Service state is ERROR, not READY"
                return super().check(url, validator)

        _FakeProcess.exit_codes["GUI"] = None
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            log = RuntimeLog(config.log_dir, console=io.StringIO())
            try:
                with (
                    patch.object(Supervisor, "_POLL_SECONDS", 0.0),
                    patch.object(Supervisor, "_RUNTIME_HEALTH_INTERVAL_SECONDS", 0.0),
                ):
                    result = Supervisor(
                        config,
                        log,
                        probe=FailingAIProbe(),
                        process_type=_FakeProcess,
                    ).run()
            finally:
                log.close()

        self.assertEqual(result, 1)
        self.assertEqual(
            _FakeProcess.events[-3:],
            ["stop:GUI", "stop:Core", "stop:AI"],
        )

    def test_unexpected_ai_exit_stops_dependents_without_restart(self) -> None:
        class CrashAfterCoreReadyProbe(_AlwaysReadyProbe):
            def check(self, url, validator):  # type: ignore[no-untyped-def]
                result = super().check(url, validator)
                if url.endswith("/api/health"):
                    _FakeProcess.exit_codes["AI"] = 17
                return result

        _FakeProcess.exit_codes["GUI"] = None
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            log = RuntimeLog(config.log_dir, console=io.StringIO())
            try:
                result = Supervisor(
                    config,
                    log,
                    probe=CrashAfterCoreReadyProbe(),
                    process_type=_FakeProcess,
                ).run()
            finally:
                log.close()

        self.assertEqual(result, 1)
        self.assertEqual(_FakeProcess.events.count("start:AI"), 1)
        self.assertEqual(
            _FakeProcess.events[-3:],
            ["stop:GUI", "stop:Core", "stop:AI"],
        )

    def test_unexpected_core_exit_stops_gui_and_ai_without_restart(self) -> None:
        class CrashAfterFinalAIProbe(_AlwaysReadyProbe):
            ai_calls = 0

            def check(self, url, validator):  # type: ignore[no-untyped-def]
                result = super().check(url, validator)
                if url.endswith("/health") and not url.endswith("/api/health"):
                    self.ai_calls += 1
                    if self.ai_calls == 2:
                        _FakeProcess.exit_codes["Core"] = 19
                return result

        _FakeProcess.exit_codes["GUI"] = None
        with tempfile.TemporaryDirectory() as directory:
            config = _config(Path(directory))
            log = RuntimeLog(config.log_dir, console=io.StringIO())
            try:
                result = Supervisor(
                    config,
                    log,
                    probe=CrashAfterFinalAIProbe(),
                    process_type=_FakeProcess,
                ).run()
            finally:
                log.close()

        self.assertEqual(result, 1)
        self.assertEqual(_FakeProcess.events.count("start:Core"), 1)
        self.assertEqual(
            _FakeProcess.events[-3:],
            ["stop:GUI", "stop:Core", "stop:AI"],
        )

    def test_health_validators_require_operational_fields(self) -> None:
        ai = _ai_health()
        self.assertIsNone(validate_ai_health(ai, require_preview=True))
        self.assertFalse(ai["tracking_running"])
        self.assertFalse(ai["latest_frame_available"])
        self.assertFalse(ai["frame_preview_published"])

        ai["camera_state"] = "ERROR"
        self.assertIsNone(validate_ai_health(ai, require_preview=True))

        ai = _ai_health()
        ai["service_state"] = "ERROR"
        self.assertIn("not READY", validate_ai_health(ai, require_preview=True) or "")

        ai = _ai_health()
        ai["yolo_model_initialized"] = False
        self.assertIn("YOLO", validate_ai_health(ai, require_preview=True) or "")

        ai = _ai_health()
        ai["frame_preview_available"] = False
        self.assertIn("unavailable", validate_ai_health(ai, require_preview=True) or "")
        self.assertIsNone(validate_ai_health(ai, require_preview=False))

        core = _core_health()
        self.assertIsNone(validate_core_health(core))
        core["ai_events_connected"] = False
        self.assertIn("not connected", validate_core_health(core) or "")


def _config(root: Path, *, windowed: bool = False) -> LauncherConfig:
    env_file = root / ".env"
    env_file.write_text(
        """SIGNATUS_AI_BASE_URL=http://127.0.0.1:8001
SIGNATUS_AI_EVENTS_URL=ws://127.0.0.1:8001/ws/events
SIGNATUS_CORE_URL=http://127.0.0.1:8000
""",
        encoding="utf-8",
    )
    return LauncherConfig.create(
        env_file=env_file,
        log_dir=Path("logs"),
        startup_timeout=1.0,
        shutdown_timeout=1.0,
        windowed=windowed,
        no_gui=False,
        inherited_environment={"TOKEN": "same"},
    )


def _ai_health() -> dict[str, object]:
    return {
        "status": "ok",
        "service_state": "READY",
        "tracking_enabled": True,
        "yolo_model_initialized": True,
        "camera_state": "STOPPED",
        "tracking_running": False,
        "latest_frame_available": False,
        "latest_frame_age_seconds": None,
        "ppe_association": "single_person_frame",
        "face_backend": "opencv_yunet_sface_fp32",
        "face_models_available": True,
        "face_models_initialized": True,
        "frame_preview_enabled": True,
        "frame_preview_available": True,
        "frame_preview_published": False,
        "frame_preview_age_seconds": None,
    }


def _core_health() -> dict[str, object]:
    return {"status": "ok", "state": "STANDBY", "ai_events_connected": True}


if __name__ == "__main__":
    unittest.main()
