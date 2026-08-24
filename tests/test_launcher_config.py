from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from signatus_launcher.config import LauncherConfig, LauncherConfigurationError


class LauncherConfigTests(unittest.TestCase):
    def test_loads_dotenv_once_with_process_environment_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / ".env"
            env_file.write_text(
                """SIGNATUS_AI_BASE_URL=http://127.0.0.1:9001
SIGNATUS_AI_EVENTS_URL=ws://127.0.0.1:9001/ws/events
SIGNATUS_CORE_URL=http://127.0.0.1:9000
FROM_FILE=present
OVERRIDDEN=file
""",
                encoding="utf-8",
            )

            config = LauncherConfig.create(
                env_file=env_file,
                log_dir=Path("logs"),
                startup_timeout=12.0,
                shutdown_timeout=4.0,
                windowed=True,
                no_gui=False,
                inherited_environment={"OVERRIDDEN": "process", "DISPLAY": ":0"},
            )

        self.assertEqual(config.project_dir, root)
        self.assertEqual(config.log_dir, root / "logs")
        self.assertEqual(config.environment["FROM_FILE"], "present")
        self.assertEqual(config.environment["OVERRIDDEN"], "process")
        self.assertEqual(config.environment["PYTHONUNBUFFERED"], "1")
        self.assertEqual(config.ai_endpoint.port, 9001)
        self.assertEqual(config.core_endpoint.port, 9000)
        self.assertTrue(config.windowed)

    def test_rejects_missing_env_file_and_nonpositive_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / ".env"
            with self.assertRaisesRegex(LauncherConfigurationError, "does not exist"):
                LauncherConfig.create(
                    env_file=missing,
                    log_dir=None,
                    startup_timeout=1.0,
                    shutdown_timeout=1.0,
                    windowed=False,
                    no_gui=True,
                    inherited_environment={},
                )

            for value in (float("nan"), float("inf")):
                with self.subTest(timeout=value), self.assertRaisesRegex(
                    LauncherConfigurationError, "startup timeout"
                ):
                    LauncherConfig.create(
                        env_file=missing,
                        log_dir=None,
                        startup_timeout=value,
                        shutdown_timeout=1.0,
                        windowed=False,
                        no_gui=True,
                        inherited_environment={},
                    )

            missing.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(LauncherConfigurationError, "startup timeout"):
                LauncherConfig.create(
                    env_file=missing,
                    log_dir=None,
                    startup_timeout=0.0,
                    shutdown_timeout=1.0,
                    windowed=False,
                    no_gui=True,
                    inherited_environment={},
                )

    def test_rejects_malformed_dotenv_and_zero_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("NOT A DOTENV LINE\n", encoding="utf-8")
            with self.assertRaisesRegex(LauncherConfigurationError, "at line 1"):
                LauncherConfig.create(
                    env_file=env_file,
                    log_dir=None,
                    startup_timeout=1.0,
                    shutdown_timeout=1.0,
                    windowed=False,
                    no_gui=True,
                    inherited_environment={},
                )

            env_file.write_text(
                """SIGNATUS_AI_BASE_URL=http://127.0.0.1:0
SIGNATUS_AI_EVENTS_URL=ws://127.0.0.1:0/ws/events
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LauncherConfigurationError, "between 1 and 65535"):
                LauncherConfig.create(
                    env_file=env_file,
                    log_dir=None,
                    startup_timeout=1.0,
                    shutdown_timeout=1.0,
                    windowed=False,
                    no_gui=True,
                    inherited_environment={},
                )

    def test_rejects_remote_or_mismatched_service_urls(self) -> None:
        cases = (
            (
                """SIGNATUS_AI_BASE_URL=http://192.0.2.1:8001
SIGNATUS_AI_EVENTS_URL=ws://192.0.2.1:8001/ws/events
""",
                "local host",
            ),
            (
                """SIGNATUS_AI_BASE_URL=http://127.0.0.1:8001
SIGNATUS_AI_EVENTS_URL=ws://127.0.0.1:9001/ws/events
""",
                "configured local AI Service",
            ),
        )
        for contents, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                env_file = Path(directory) / ".env"
                env_file.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(LauncherConfigurationError, message):
                    LauncherConfig.create(
                        env_file=env_file,
                        log_dir=None,
                        startup_timeout=1.0,
                        shutdown_timeout=1.0,
                        windowed=False,
                        no_gui=True,
                        inherited_environment={},
                    )


if __name__ == "__main__":
    unittest.main()
