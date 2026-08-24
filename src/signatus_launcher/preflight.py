from __future__ import annotations

import importlib.util
import math
import os
import socket
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from signatus_ai.tracker import validate_model_class_names
from signatus_contracts.frame_buffer import FrameBufferContractError, segment_size
from signatus_core.worksites import WorksiteRepository

from .config import LauncherConfig

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_APPROVED_FACE_THRESHOLD = 0.35
_APPROVED_PERSON_CLASS = "Person"
_APPROVED_PPE_ASSOCIATION = "single_person_frame"


@dataclass(slots=True)
class PreflightReport:
    # ``errors`` remains the compatibility name for deployment-wide fatal
    # errors. Data errors have their own non-blocking collection.
    errors: list[str] = field(default_factory=list)
    data_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def fatal_errors(self) -> list[str]:
        return self.errors

    @property
    def outcome(self) -> PreflightOutcome:
        if self.errors:
            return PreflightOutcome.FAIL
        if self.data_errors:
            return PreflightOutcome.PASS_WITH_DATA_ERRORS
        return PreflightOutcome.PASS

    def error(self, message: str) -> None:
        self.errors.append(message)

    fatal = error

    def data_error(self, message: str) -> None:
        self.data_errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class PreflightOutcome(StrEnum):
    PASS = "PASS"
    PASS_WITH_DATA_ERRORS = "PASS WITH DATA ERRORS"
    FAIL = "FAIL"


def run_preflight(config: LauncherConfig, *, include_gui: bool) -> PreflightReport:
    """Perform static, non-camera-opening checks for an operational launch."""

    report = PreflightReport()
    environment = config.environment

    _check_python_modules(report, include_gui=include_gui)
    _check_approved_settings(report, config)
    _check_model_paths(report, config)
    _check_worksites(report, config)
    _check_camera(report, config)
    _check_ports(report, config)
    _check_shared_memory(report, config, include_gui=include_gui)

    if include_gui and not _display_available(environment):
        report.error("no graphical display is available (DISPLAY or WAYLAND_DISPLAY is unset)")
    return report


def _check_python_modules(report: PreflightReport, *, include_gui: bool) -> None:
    modules = [
        "cv2",
        "numpy",
        "openvino",
        "ultralytics",
        "uvicorn",
        "yaml",
        "signatus_ai",
        "signatus_core",
    ]
    if include_gui:
        modules.extend(("PySide6", "signatus_gui"))
    for module in modules:
        if importlib.util.find_spec(module) is None:
            report.error(f"required Python module is not installed: {module}")


def _check_approved_settings(report: PreflightReport, config: LauncherConfig) -> None:
    environment = config.environment
    tracking = _parse_bool(
        report,
        "SIGNATUS_AI_TRACKING_ENABLED",
        environment.get("SIGNATUS_AI_TRACKING_ENABLED", "false"),
    )
    if tracking is False:
        report.error("SIGNATUS_AI_TRACKING_ENABLED must be true for an operational launch")

    frame_preview = _parse_bool(
        report,
        "SIGNATUS_FRAME_SHM_ENABLED",
        environment.get("SIGNATUS_FRAME_SHM_ENABLED", "true"),
    )
    if not config.no_gui and frame_preview is False:
        report.error("SIGNATUS_FRAME_SHM_ENABLED must be true when launching the GUI")

    person_class = environment.get("SIGNATUS_PERSON_CLASS", _APPROVED_PERSON_CLASS)
    if person_class != _APPROVED_PERSON_CLASS:
        report.error(
            f"SIGNATUS_PERSON_CLASS must remain the approved exact class {_APPROVED_PERSON_CLASS!r}"
        )

    association = environment.get("SIGNATUS_PPE_ASSOCIATION", _APPROVED_PPE_ASSOCIATION)
    if association != _APPROVED_PPE_ASSOCIATION:
        report.error(
            "SIGNATUS_PPE_ASSOCIATION must remain the approved single_person_frame strategy"
        )

    threshold = _parse_float(
        report,
        "SIGNATUS_FACE_MATCH_MIN_COSINE_SIMILARITY",
        environment.get("SIGNATUS_FACE_MATCH_MIN_COSINE_SIMILARITY", "0.35"),
    )
    if threshold is not None and threshold != _APPROVED_FACE_THRESHOLD:
        report.error(
            "SIGNATUS_FACE_MATCH_MIN_COSINE_SIMILARITY must remain the owner-approved "
            "functional value 0.35 until calibration is complete"
        )

    _check_positive_float(
        report,
        "SIGNATUS_TRACK_LOST_TIMEOUT_SECONDS",
        environment.get("SIGNATUS_TRACK_LOST_TIMEOUT_SECONDS", "1.5"),
    )
    _check_positive_float(
        report,
        "SIGNATUS_FRAME_STALE_SECONDS",
        environment.get("SIGNATUS_FRAME_STALE_SECONDS", "3.0"),
    )

    capacity_value = environment.get("SIGNATUS_PREVIEW_MAX_FRAME_BYTES", "6220800")
    try:
        capacity = int(capacity_value)
        segment_size(capacity)
    except (ValueError, FrameBufferContractError):
        report.error("SIGNATUS_PREVIEW_MAX_FRAME_BYTES must be a valid positive slot capacity")

    name = environment.get("SIGNATUS_FRAME_SHM_NAME", "signatus_camera_v1").strip()
    if frame_preview is True and not name:
        report.error("SIGNATUS_FRAME_SHM_NAME must not be empty when preview is enabled")


def _check_model_paths(report: PreflightReport, config: LauncherConfig) -> None:
    environment = config.environment
    model_path = config.resolve_runtime_path(
        environment.get("SIGNATUS_MODEL_PATH", "./models/yolo26s100e18b_int8_openvino_model")
    )
    if not model_path.is_dir():
        report.error(f"OpenVINO model directory does not exist: {model_path}")
    else:
        xml_files = tuple(model_path.glob("*.xml"))
        bin_files = tuple(model_path.glob("*.bin"))
        if len(xml_files) != 1 or len(bin_files) != 1:
            report.error(
                f"OpenVINO model directory must contain exactly one .xml and one .bin file: "
                f"{model_path}"
            )
        _check_model_metadata(report, model_path)

    face_paths = (
        (
            "YuNet face detector model",
            environment.get(
                "SIGNATUS_FACE_DETECTOR_MODEL_PATH",
                "./models/face_detection_yunet_2023mar.onnx",
            ),
        ),
        (
            "SFace recognizer model",
            environment.get(
                "SIGNATUS_FACE_RECOGNIZER_MODEL_PATH",
                "./models/face_recognition_sface_2021dec.onnx",
            ),
        ),
    )
    for label, value in face_paths:
        path = config.resolve_runtime_path(value)
        if not path.is_file():
            report.error(f"{label} does not exist: {path}")


def _check_model_metadata(report: PreflightReport, model_path: Path) -> None:
    metadata_path = model_path / "metadata.yaml"
    if not metadata_path.is_file():
        report.error(f"OpenVINO model metadata does not exist: {metadata_path}")
        return
    try:
        import yaml
    except ModuleNotFoundError:
        # The dependency check already reports PyYAML without hiding other
        # independent preflight errors.
        return
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        names = metadata["names"]
        validate_model_class_names(names)
    except (KeyError, TypeError, ValueError, OSError, UnicodeError, yaml.YAMLError) as exc:
        report.error(f"OpenVINO model metadata class policy is invalid: {exc}")


def _check_worksites(report: PreflightReport, config: LauncherConfig) -> None:
    value = config.environment.get("SIGNATUS_WORKSITE_DIR", "./config/worksites")
    directory = config.resolve_runtime_path(value)
    if not directory.is_dir():
        report.error(f"worksite directory does not exist: {directory}")
        return

    try:
        catalog = WorksiteRepository(directory).load_catalog()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report.error(f"Core could not validate worksite configuration: {exc}")
        return
    _merge_core_validation(report, catalog.validation_report)


def _merge_core_validation(report: PreflightReport, validation_report: object) -> None:
    issues = getattr(validation_report, "issues", None)
    if not isinstance(issues, (list, tuple)):
        report.error("Core returned an invalid deployment-data validation report")
        return
    for issue in issues:
        severity = getattr(getattr(issue, "severity", None), "value", None)
        message = getattr(issue, "message", None)
        if not isinstance(message, str) or not message:
            report.error("Core returned a malformed deployment-data validation issue")
            continue
        context: list[str] = []
        worksite_id = getattr(issue, "worksite_id", None)
        worker_id = getattr(issue, "worker_id", None)
        source = getattr(issue, "source", None)
        if isinstance(worksite_id, str) and worksite_id:
            context.append(f"Wo.No. {worksite_id}")
        if isinstance(worker_id, str) and worker_id:
            context.append(f"Worker {worker_id}")
        if not context and isinstance(source, str) and source:
            context.append(source)
        rendered = f"{' / '.join(context)}: {message}" if context else message
        if severity == "FATAL":
            report.error(rendered)
        elif severity == "DATA_ERROR":
            report.data_error(rendered)
        elif severity == "WARNING":
            report.warning(rendered)
        else:
            report.error(f"Core returned an unknown validation severity for: {rendered}")


def _check_camera(report: PreflightReport, config: LauncherConfig) -> None:
    raw_source = config.environment.get("SIGNATUS_CAMERA_SOURCE", "0").strip()
    if not raw_source:
        report.warning(
            "SIGNATUS_CAMERA_SOURCE is empty; Start Camera will remain unavailable until configured"
        )
        return
    if raw_source.isdecimal():
        path = Path(f"/dev/video{int(raw_source)}")
    else:
        candidate = Path(raw_source).expanduser()
        if not candidate.is_absolute():
            report.warning(
                "SIGNATUS_CAMERA_SOURCE does not name an absolute local device path; "
                "Start Camera may fail"
            )
            return
        path = candidate
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        report.warning(
            f"camera device is currently unavailable: {path} ({exc.strerror or exc}); "
            "Signatus will start with Camera STOPPED"
        )
        return
    if not stat.S_ISCHR(mode):
        report.warning(
            f"configured camera source is not a character device: {path}; Start Camera may fail"
        )
    if not os.access(path, os.R_OK | os.W_OK):
        report.warning(
            f"camera device is not readable and writable by the current user: {path}; "
            "Start Camera may fail"
        )


def _check_ports(report: PreflightReport, config: LauncherConfig) -> None:
    if config.ai_endpoint.port == config.core_endpoint.port:
        report.error("AI Service and Core cannot use the same TCP port")
        return
    ai_bind_host = "::1" if config.ai_endpoint.host == "::1" else "127.0.0.1"
    core_bind_host = "::1" if config.core_endpoint.host == "::1" else "0.0.0.0"
    if not _port_available(ai_bind_host, config.ai_endpoint.port):
        report.error(
            f"AI Service port is already in use: "
            f"{config.ai_endpoint.host}:{config.ai_endpoint.port}"
        )
    if not _port_available(core_bind_host, config.core_endpoint.port):
        report.error(
            f"Core port is already in use: {core_bind_host}:{config.core_endpoint.port}"
        )


def _check_shared_memory(
    report: PreflightReport,
    config: LauncherConfig,
    *,
    include_gui: bool,
) -> None:
    enabled = _bool_value(config.environment.get("SIGNATUS_FRAME_SHM_ENABLED", "true"))
    if enabled is not True:
        return
    name = config.environment.get("SIGNATUS_FRAME_SHM_NAME", "signatus_camera_v1").strip()
    if not name:
        return
    normalized_name = name.removeprefix("/")
    if not normalized_name or "/" in normalized_name or len(normalized_name.encode()) > 255:
        report.error("SIGNATUS_FRAME_SHM_NAME is not a valid POSIX shared-memory name")
        return
    shared_memory_path = Path("/dev/shm") / normalized_name
    if shared_memory_path.exists():
        report.error(
            f"shared-memory preview name is already in use: {name}; "
            "the launcher will not unlink it"
        )
    if include_gui and not Path("/dev/shm").is_dir():
        report.error("the local shared-memory filesystem /dev/shm is unavailable")


def _port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    bind_host = host
    try:
        with socket.socket(family, socket.SOCK_STREAM) as candidate:
            # Match Uvicorn/asyncio's listener behavior so a recently closed
            # connection in TIME_WAIT is not misreported as a live port owner.
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            candidate.bind((bind_host, port))
    except OSError:
        return False
    return True


def _display_available(environment: Mapping[str, str]) -> bool:
    get = environment.get
    direct_backends = {"eglfs", "linuxfb", "vkkhrdisplay"}
    return bool(
        get("DISPLAY")
        or get("WAYLAND_DISPLAY")
        or str(get("QT_QPA_PLATFORM", "")).casefold() in direct_backends
    )


def _parse_bool(report: PreflightReport, variable: str, value: str) -> bool | None:
    parsed = _bool_value(value)
    if parsed is None:
        report.error(f"{variable} must be one of true/false, yes/no, on/off, or 1/0")
    return parsed


def _bool_value(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _parse_float(report: PreflightReport, variable: str, value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        report.error(f"{variable} must be a finite number")
        return None
    if not math.isfinite(parsed):
        report.error(f"{variable} must be a finite number")
        return None
    return parsed


def _check_positive_float(report: PreflightReport, variable: str, value: str) -> None:
    parsed = _parse_float(report, variable, value)
    if parsed is not None and parsed <= 0:
        report.error(f"{variable} must be positive")
