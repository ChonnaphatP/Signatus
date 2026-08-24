from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import FrameType
from typing import TextIO
from uuid import uuid4

from .config import LauncherConfig


class LauncherRuntimeError(RuntimeError):
    """Raised when the supervised stack cannot remain operational."""


class RuntimeLog:
    def __init__(
        self,
        log_dir: Path,
        *,
        console: TextIO | None = None,
        launch_id: str | None = None,
    ) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "signatus.log"
        self.launch_id = launch_id or uuid4().hex
        self._logger = logging.getLogger(f"signatus-launch.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        console_handler = logging.StreamHandler(console or sys.stdout)
        console_handler.setFormatter(logging.Formatter("[%(component)s] %(message)s"))
        self._logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            self.path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(
            "%(asctime)sZ [launch=%(launch_id)s] [%(component)s] "
            "[pid=%(pid)s] [exit=%(exit_code)s] "
            "[severity=%(validation_severity)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        file_formatter.converter = time.gmtime
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)

    def info(
        self,
        component: str,
        message: str,
        *,
        pid: int | None = None,
        exit_code: int | None = None,
        validation_severity: str | None = None,
    ) -> None:
        self._write(
            logging.INFO,
            component,
            message,
            pid=pid,
            exit_code=exit_code,
            validation_severity=validation_severity,
        )

    def warning(
        self,
        component: str,
        message: str,
        *,
        pid: int | None = None,
        exit_code: int | None = None,
        validation_severity: str | None = None,
    ) -> None:
        self._write(
            logging.WARNING,
            component,
            message,
            pid=pid,
            exit_code=exit_code,
            validation_severity=validation_severity,
        )

    def error(
        self,
        component: str,
        message: str,
        *,
        pid: int | None = None,
        exit_code: int | None = None,
        validation_severity: str | None = None,
    ) -> None:
        self._write(
            logging.ERROR,
            component,
            message,
            pid=pid,
            exit_code=exit_code,
            validation_severity=validation_severity,
        )

    def _write(
        self,
        level: int,
        component: str,
        message: str,
        *,
        pid: int | None,
        exit_code: int | None,
        validation_severity: str | None,
    ) -> None:
        self._logger.log(
            level,
            message,
            extra={
                "component": component,
                "launch_id": self.launch_id,
                "pid": os.getpid() if pid is None else pid,
                "exit_code": "-" if exit_code is None else exit_code,
                "validation_severity": validation_severity or "-",
            },
        )

    def close(self) -> None:
        for handler in tuple(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)


class ManagedProcess:
    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        runtime_log: RuntimeLog,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.name = name
        self.command = tuple(command)
        self._cwd = cwd
        self._environment = dict(environment)
        self._log = runtime_log
        self._popen = popen
        self._process: subprocess.Popen[str] | None = None
        self._output_thread: threading.Thread | None = None
        self._exit_logged = False

    @property
    def started(self) -> bool:
        return self._process is not None

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def start(self) -> None:
        if self._process is not None:
            raise LauncherRuntimeError(f"{self.name} was already started")
        self._log.info("launcher", f"starting {self.name}")
        try:
            process = self._popen(
                list(self.command),
                cwd=self._cwd,
                env=self._environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise LauncherRuntimeError(f"could not start {self.name}: {exc}") from exc
        self._process = process
        self._log.info("launcher", f"started {self.name}", pid=process.pid)
        self._output_thread = threading.Thread(
            target=self._drain_output,
            name=f"{self.name}-log-reader",
            daemon=True,
        )
        self._output_thread.start()

    def poll(self) -> int | None:
        process = self._process
        if process is None:
            return None
        result = process.poll()
        if result is not None and not self._exit_logged:
            self._exit_logged = True
            self._log.info(
                "launcher",
                f"{self.name} process exited",
                pid=process.pid,
                exit_code=result,
            )
        return result

    def stop(self, timeout: float) -> bool:
        process = self._process
        if process is None:
            return True

        stages = (
            (signal.SIGINT, "requesting graceful shutdown"),
            (signal.SIGTERM, "terminating"),
            (signal.SIGKILL, "force-killing"),
        )
        stopped = not self._group_exists()
        for signum, action in stages:
            if stopped:
                break
            self._log.info("launcher", f"{action} {self.name}", pid=process.pid)
            self._signal_group(signum)
            stopped = self._wait_for_group_exit(timeout)
            if not stopped and signum != signal.SIGKILL:
                self._log.warning(
                    "launcher",
                    f"{self.name} did not stop within {timeout:g}s after {signum.name}",
                    pid=process.pid,
                )

        result = self.poll()
        if not stopped:
            self._log.error(
                "launcher",
                f"{self.name} process group survived SIGKILL",
                pid=process.pid,
                exit_code=result,
            )
        if self._output_thread is not None:
            self._output_thread.join(timeout=1.0)
        return stopped

    def _signal_group(self, signum: signal.Signals) -> None:
        process = self._process
        if process is None:
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    def _group_exists(self) -> bool:
        process = self._process
        if process is None:
            return False
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _wait_for_group_exit(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            self.poll()
            if not self._group_exists():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _drain_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self._log.info(self.name, line.rstrip("\r\n"), pid=process.pid)
        except (OSError, ValueError) as exc:
            self._log.warning("launcher", f"stopped reading {self.name} output: {exc}")
        finally:
            process.stdout.close()


class HealthProbe:
    def __init__(self, *, request_timeout: float = 3.0) -> None:
        self._request_timeout = request_timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def check(
        self,
        url: str,
        validator: Callable[[Mapping[str, object]], str | None],
    ) -> tuple[bool, str]:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with self._opener.open(request, timeout=self._request_timeout) as response:
                if response.status != 200:
                    return False, f"HTTP {response.status}"
                payload = json.load(response)
        except (
            OSError,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            json.JSONDecodeError,
            UnicodeError,
        ) as exc:
            return False, str(exc)
        if not isinstance(payload, dict):
            return False, "health response is not a JSON object"
        problem = validator(payload)
        return (problem is None, "ready" if problem is None else problem)


def validate_ai_health(
    payload: Mapping[str, object],
    *,
    require_preview: bool,
    max_frame_age_seconds: float = 10.0,
) -> str | None:
    # Retain the argument for source compatibility with callers from the first
    # supervisor version. Camera-frame age is intentionally not service health.
    del max_frame_age_seconds
    if payload.get("status") != "ok":
        return "AI status is not ok"
    service_state = payload.get("service_state")
    if service_state != "READY":
        return f"AI Service state is {service_state or 'unknown'}, not READY"
    if payload.get("tracking_enabled") is not True:
        return "AI tracking is disabled"
    if payload.get("yolo_model_initialized") is not True:
        return "AI YOLO model is not initialized"
    if payload.get("ppe_association") != "single_person_frame":
        return "AI PPE association is not approved"
    if payload.get("face_backend") != "opencv_yunet_sface_fp32":
        return "AI face backend is not approved"
    if payload.get("face_models_available") is not True:
        return "AI face model files are unavailable"
    if payload.get("face_models_initialized") is not True:
        return "AI face models are not initialized"
    camera_state = payload.get("camera_state")
    if camera_state not in {"STOPPED", "STARTING", "RUNNING", "STOPPING", "ERROR"}:
        return "AI camera state is invalid"
    if require_preview:
        if payload.get("frame_preview_enabled") is not True:
            return "AI shared-memory preview is disabled"
        if payload.get("frame_preview_available") is not True:
            return "AI shared-memory preview is unavailable"
    return None


def validate_core_health(payload: Mapping[str, object]) -> str | None:
    if payload.get("status") != "ok":
        return "Core status is not ok"
    if payload.get("state") not in {"STANDBY", "AUTHORIZATION"}:
        return "Core returned an unknown screening state"
    if payload.get("ai_events_connected") is not True:
        return "Core is not connected to AI tracking events"
    return None


class Supervisor:
    _POLL_SECONDS = 0.25
    _RUNTIME_HEALTH_INTERVAL_SECONDS = 1.0
    _RUNTIME_HEALTH_FAILURE_LIMIT = 3

    def __init__(
        self,
        config: LauncherConfig,
        runtime_log: RuntimeLog,
        *,
        probe: HealthProbe | None = None,
        process_type: type[ManagedProcess] = ManagedProcess,
    ) -> None:
        self._config = config
        self._log = runtime_log
        self._probe = probe or HealthProbe()
        self._process_type = process_type
        self._processes: list[ManagedProcess] = []
        self._child_environment = dict(config.environment)
        self._child_environment["SIGNATUS_LAUNCH_ID"] = runtime_log.launch_id
        self._child_environment["SIGNATUS_LAUNCHER_PID"] = str(os.getpid())
        self._received_signal: int | None = None
        self._previous_handlers: dict[int, signal.Handlers] = {}

    def run(self) -> int:
        self._install_signal_handlers()
        result = 1
        try:
            ai = self._start("AI", self._ai_command())
            self._wait_until_ready(
                ai,
                "AI Service",
                self._ai_health_url(),
                lambda payload: validate_ai_health(
                    payload,
                    require_preview=not self._config.no_gui,
                ),
            )

            core = self._start("Core", self._core_command())
            self._wait_until_ready(
                core,
                "Core",
                self._core_health_url(),
                validate_core_health,
            )

            ai_ready, ai_detail = self._probe.check(
                self._ai_health_url(),
                lambda payload: validate_ai_health(
                    payload,
                    require_preview=not self._config.no_gui,
                ),
            )
            if not ai_ready:
                raise LauncherRuntimeError(
                    f"AI Service lost readiness before GUI startup: {ai_detail}"
                )

            gui: ManagedProcess | None = None
            if not self._config.no_gui:
                gui = self._start("GUI", self._gui_command())
            self._log.info("launcher", "Signatus is operational")
            result = self._monitor(ai, core, gui)
        except LauncherRuntimeError as exc:
            self._log.error("launcher", str(exc))
            result = 1 if self._received_signal is None else 128 + self._received_signal
        finally:
            shutdown_ok = self._shutdown()
            self._restore_signal_handlers()
        return 1 if result == 0 and not shutdown_ok else result

    def _start(self, name: str, command: Sequence[str]) -> ManagedProcess:
        process = self._process_type(
            name,
            command,
            cwd=self._config.project_dir,
            environment=self._child_environment,
            runtime_log=self._log,
        )
        self._processes.append(process)
        process.start()
        return process

    def _wait_until_ready(
        self,
        process: ManagedProcess,
        display_name: str,
        url: str,
        validator: Callable[[Mapping[str, object]], str | None],
    ) -> None:
        deadline = time.monotonic() + self._config.startup_timeout
        last_detail: str | None = None
        while time.monotonic() < deadline:
            self._raise_if_interrupted()
            self._raise_if_any_process_exited(during=f"{display_name} startup")
            ready, detail = self._probe.check(url, validator)
            if ready:
                self._log.info("launcher", f"{display_name} is ready")
                return
            if detail != last_detail:
                self._log.info("launcher", f"waiting for {display_name}: {detail}")
                last_detail = detail
            if display_name == "AI Service" and detail in {
                "AI Service state is ERROR, not READY",
                "AI tracking is disabled",
                "AI YOLO model is not initialized",
                "AI PPE association is not approved",
                "AI face backend is not approved",
                "AI face model files are unavailable",
                "AI face models are not initialized",
                "AI shared-memory preview is disabled",
                "AI shared-memory preview is unavailable",
            }:
                raise LauncherRuntimeError(f"{display_name} cannot become ready: {detail}")
            time.sleep(self._POLL_SECONDS)
        self._raise_if_any_process_exited(during=f"{display_name} startup")
        raise LauncherRuntimeError(
            f"{display_name} did not become ready within "
            f"{self._config.startup_timeout:g}s: {last_detail or 'no health response'}"
        )

    def _monitor(
        self,
        ai: ManagedProcess,
        core: ManagedProcess,
        gui: ManagedProcess | None,
    ) -> int:
        next_health_check = time.monotonic()
        failures = {"AI": 0, "Core": 0}
        while True:
            if self._received_signal is not None:
                return 128 + self._received_signal
            self._raise_if_process_exited(ai, during="operation")
            self._raise_if_process_exited(core, during="operation")
            if gui is not None:
                gui_result = gui.poll()
                if gui_result is not None:
                    if gui_result == 0:
                        self._log.info("launcher", "GUI exited normally")
                        return 0
                    raise LauncherRuntimeError(f"GUI exited unexpectedly with status {gui_result}")

            now = time.monotonic()
            if now >= next_health_check:
                checks = (
                    (
                        "AI",
                        self._ai_health_url(),
                        lambda payload: validate_ai_health(
                            payload,
                            require_preview=not self._config.no_gui,
                        ),
                    ),
                    ("Core", self._core_health_url(), validate_core_health),
                )
                for name, url, validator in checks:
                    ready, detail = self._probe.check(url, validator)
                    failures[name] = 0 if ready else failures[name] + 1
                    if not ready:
                        self._log.warning(
                            "launcher",
                            f"{name} readiness check failed "
                            f"({failures[name]}/{self._RUNTIME_HEALTH_FAILURE_LIMIT}): {detail}",
                        )
                    if failures[name] >= self._RUNTIME_HEALTH_FAILURE_LIMIT:
                        raise LauncherRuntimeError(
                            f"{name} lost operational readiness: {detail}"
                        )
                next_health_check = now + self._RUNTIME_HEALTH_INTERVAL_SECONDS
            time.sleep(self._POLL_SECONDS)

    def _raise_if_interrupted(self) -> None:
        if self._received_signal is not None:
            raise LauncherRuntimeError(
                f"startup interrupted by signal {self._received_signal}"
            )

    @staticmethod
    def _raise_if_process_exited(process: ManagedProcess, *, during: str) -> None:
        result = process.poll()
        if result is not None:
            raise LauncherRuntimeError(
                f"{process.name} exited with status {result} during {during}"
            )

    def _raise_if_any_process_exited(self, *, during: str) -> None:
        for process in self._processes:
            self._raise_if_process_exited(process, during=during)

    def _shutdown(self) -> bool:
        all_stopped = True
        for process in reversed(self._processes):
            try:
                result = process.stop(self._config.shutdown_timeout)
                if result is False:
                    all_stopped = False
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                all_stopped = False
                self._log.error("launcher", f"could not stop {process.name}: {exc}")
        if self._processes and all_stopped:
            self._log.info("launcher", "all Signatus processes have stopped")
        elif self._processes:
            self._log.error("launcher", "one or more Signatus process groups did not stop cleanly")
        return all_stopped

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        if self._received_signal is None:
            self._received_signal = signum
            self._log.info("launcher", f"received signal {signum}; stopping Signatus")

    def _ai_command(self) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            "uvicorn",
            "signatus_ai.app:app",
            "--host",
            self._loopback_bind_host(self._config.ai_endpoint.host),
            "--port",
            str(self._config.ai_endpoint.port),
        )

    def _core_command(self) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            "uvicorn",
            "signatus_core.app:app",
            "--host",
            "::1" if self._config.core_endpoint.host == "::1" else "0.0.0.0",
            "--port",
            str(self._config.core_endpoint.port),
        )

    def _gui_command(self) -> tuple[str, ...]:
        command = [sys.executable, "-m", "signatus_gui"]
        if self._config.windowed:
            command.append("--windowed")
        return tuple(command)

    def _ai_health_url(self) -> str:
        base = self._config.environment.get(
            "SIGNATUS_AI_BASE_URL", "http://127.0.0.1:8001"
        )
        return f"{base.rstrip('/')}/health"

    def _core_health_url(self) -> str:
        base = self._config.environment.get("SIGNATUS_CORE_URL", "http://127.0.0.1:8000")
        return f"{base.rstrip('/')}/api/health"

    @staticmethod
    def _loopback_bind_host(host: str) -> str:
        return "::1" if host == "::1" else "127.0.0.1"
