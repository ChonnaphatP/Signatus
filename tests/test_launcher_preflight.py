from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from signatus_ai.tracker import APPROVED_MODEL_CLASS_NAMES
from signatus_launcher.config import LauncherConfig
from signatus_launcher.preflight import run_preflight

_TEST_METADATA = "names:\n" + "".join(
    f"  {index}: {name}\n" for index, name in APPROVED_MODEL_CLASS_NAMES.items()
)
_TEST_MODEL_SHA256 = {
    "OpenVINO model XML": hashlib.sha256(b"xml").hexdigest(),
    "OpenVINO model BIN": hashlib.sha256(b"bin").hexdigest(),
    "OpenVINO model metadata": hashlib.sha256(_TEST_METADATA.encode()).hexdigest(),
    "YuNet face detector model": hashlib.sha256(b"model").hexdigest(),
    "SFace recognizer model": hashlib.sha256(b"model").hexdigest(),
}


class LauncherPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        checksum_patch = patch.dict(
            "signatus_launcher.preflight._EXPECTED_MODEL_SHA256",
            _TEST_MODEL_SHA256,
            clear=True,
        )
        checksum_patch.start()
        self.addCleanup(checksum_patch.stop)

    def test_accepts_complete_approved_static_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _make_config(Path(directory))
            with patch("signatus_launcher.preflight._port_available", return_value=True):
                report = run_preflight(config, include_gui=True)

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.outcome.value, "PASS")
        self.assertEqual(report.warnings, [])

    def test_rejects_parked_settings_and_example_worksite_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overrides = {
                "SIGNATUS_AI_TRACKING_ENABLED": "false",
                "SIGNATUS_PERSON_CLASS": "person",
                "SIGNATUS_PPE_ASSOCIATION": "iou",
                "SIGNATUS_FACE_MATCH_MIN_COSINE_SIMILARITY": "0.5",
                "SIGNATUS_FRAME_SHM_ENABLED": "false",
            }
            config = _make_config(root, overrides=overrides, example_worksite=True)
            with patch("signatus_launcher.preflight._port_available", return_value=True):
                report = run_preflight(config, include_gui=True)

        fatal = "\n".join(report.errors)
        self.assertIn("TRACKING_ENABLED must be true", fatal)
        self.assertIn("exact class 'Person'", fatal)
        self.assertIn("single_person_frame", fatal)
        self.assertIn("functional value 0.35", fatal)
        self.assertIn("SHM_ENABLED must be true", fatal)
        self.assertTrue(report.data_errors)

    def test_rejects_unknown_ppe_and_invalid_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _make_config(
                root,
                worksite={
                    "worksite_id": "WO-001",
                    "name": "Gate",
                    "required_ppe": ["respirator"],
                    "authorized_workers": [
                        {"worker_id": "EMP1", "name": "Employee 1", "embedding": [0.0, 0.0]},
                        {"worker_id": "EMP2", "name": "Employee 2", "embedding": [1.0]},
                    ],
                },
            )
            with patch("signatus_launcher.preflight._port_available", return_value=True):
                report = run_preflight(config, include_gui=True)

        combined = "\n".join(report.data_errors)
        self.assertIn("no approved Core policy", combined)
        self.assertIn("Expected 128 values", combined)
        self.assertEqual(report.errors, [])

    def test_missing_camera_is_a_warning_and_does_not_block_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _make_config(
                Path(directory),
                overrides={"SIGNATUS_CAMERA_SOURCE": "/dev/signatus-missing-camera"},
            )
            with patch("signatus_launcher.preflight._port_available", return_value=True):
                report = run_preflight(config, include_gui=True)

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.outcome.value, "PASS")
        self.assertTrue(any("Camera STOPPED" in item for item in report.warnings))

    def test_invalid_embedding_is_nonfatal_data_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _make_config(
                Path(directory),
                worksite={
                    "worksite_id": "WO-001",
                    "name": "Gate",
                    "required_ppe": ["helmet"],
                    "authorized_workers": [
                        {
                            "worker_id": "EMP1",
                            "name": "Employee 1",
                            "embedding": [1.0, 0.0, 0.0],
                        },
                    ],
                },
            )
            with patch("signatus_launcher.preflight._port_available", return_value=True):
                report = run_preflight(config, include_gui=True)

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.outcome.value, "PASS WITH DATA ERRORS")
        self.assertTrue(any("Expected 128 values" in item for item in report.data_errors))
        self.assertTrue(any("Worker EMP1" in item for item in report.data_errors))

    def test_rejects_wrong_model_class_mapping_and_shared_memory_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _make_config(
                root,
                overrides={"SIGNATUS_FRAME_SHM_NAME": "invalid/name"},
            )
            metadata = root / "models" / "detector_openvino_model" / "metadata.yaml"
            metadata.write_text(
                metadata.read_text(encoding="utf-8").replace("7: no_helmet", "7: helmet"),
                encoding="utf-8",
            )
            with patch("signatus_launcher.preflight._port_available", return_value=True):
                report = run_preflight(config, include_gui=True)

        combined = "\n".join(report.errors)
        self.assertIn("class policy is invalid", combined)
        self.assertIn("valid POSIX shared-memory name", combined)

    def test_rejects_model_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _make_config(root)
            (root / "models" / "sface.onnx").write_bytes(b"wrong model")
            with patch("signatus_launcher.preflight._port_available", return_value=True):
                report = run_preflight(config, include_gui=True)

        self.assertTrue(
            any("SFace recognizer model checksum mismatch" in item for item in report.errors)
        )

    def test_reports_occupied_ports_without_starting_or_reusing_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _make_config(Path(directory))
            with patch("signatus_launcher.preflight._port_available", return_value=False):
                report = run_preflight(config, include_gui=True)

        self.assertTrue(any("AI Service port is already in use" in item for item in report.errors))
        self.assertTrue(any("Core port is already in use" in item for item in report.errors))
        self.assertEqual(report.outcome.value, "FAIL")


def _make_config(
    root: Path,
    *,
    overrides: dict[str, str] | None = None,
    example_worksite: bool = False,
    worksite: dict[str, object] | None = None,
) -> LauncherConfig:
    model_dir = root / "models" / "detector_openvino_model"
    model_dir.mkdir(parents=True)
    (model_dir / "detector.xml").write_text("xml", encoding="utf-8")
    (model_dir / "detector.bin").write_bytes(b"bin")
    (model_dir / "metadata.yaml").write_text(_TEST_METADATA, encoding="utf-8")
    (root / "models" / "yunet.onnx").write_bytes(b"model")
    (root / "models" / "sface.onnx").write_bytes(b"model")

    worksite_dir = root / "config" / "worksites"
    worksite_dir.mkdir(parents=True)
    payload = worksite or {
        "worksite_id": "WO-001",
        "name": "Gate",
        "required_ppe": ["helmet"],
        "authorized_workers": [
            {"worker_id": "EMP1", "name": "Employee 1", "embedding": [1.0] + [0.0] * 127}
        ],
    }
    name = "WO-001.example.json" if example_worksite else "WO-001.json"
    (worksite_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    values = {
        "SIGNATUS_AI_BASE_URL": "http://127.0.0.1:39081",
        "SIGNATUS_AI_EVENTS_URL": "ws://127.0.0.1:39081/ws/events",
        "SIGNATUS_CORE_URL": "http://127.0.0.1:39080",
        "SIGNATUS_WORKSITE_DIR": "./config/worksites",
        "SIGNATUS_FACE_MATCH_MIN_COSINE_SIMILARITY": "0.35",
        "SIGNATUS_MODEL_PATH": "./models/detector_openvino_model",
        "SIGNATUS_CAMERA_SOURCE": "/dev/null",
        "SIGNATUS_PERSON_CLASS": "Person",
        "SIGNATUS_TRACK_LOST_TIMEOUT_SECONDS": "1.5",
        "SIGNATUS_AI_TRACKING_ENABLED": "true",
        "SIGNATUS_PPE_ASSOCIATION": "single_person_frame",
        "SIGNATUS_FRAME_SHM_ENABLED": "true",
        "SIGNATUS_FRAME_SHM_NAME": f"signatus_test_{root.name}",
        "SIGNATUS_PREVIEW_MAX_FRAME_BYTES": "6220800",
        "SIGNATUS_FRAME_STALE_SECONDS": "3.0",
        "SIGNATUS_FACE_DETECTOR_MODEL_PATH": "./models/yunet.onnx",
        "SIGNATUS_FACE_RECOGNIZER_MODEL_PATH": "./models/sface.onnx",
    }
    values.update(overrides or {})
    env_file = root / ".env"
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()),
        encoding="utf-8",
    )
    return LauncherConfig.create(
        env_file=env_file,
        log_dir=None,
        startup_timeout=1.0,
        shutdown_timeout=1.0,
        windowed=True,
        no_gui=False,
        inherited_environment={"DISPLAY": ":99"},
    )


if __name__ == "__main__":
    unittest.main()
