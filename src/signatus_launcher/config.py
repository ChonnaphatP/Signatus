from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import SplitResult, urlsplit

from dotenv import dotenv_values
from dotenv.parser import parse_stream


class LauncherConfigurationError(ValueError):
    """Raised when launcher arguments or the dotenv file are unusable."""


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class LauncherConfig:
    project_dir: Path
    env_file: Path
    log_dir: Path
    environment: Mapping[str, str]
    ai_endpoint: ServiceEndpoint
    core_endpoint: ServiceEndpoint
    startup_timeout: float
    shutdown_timeout: float
    windowed: bool
    no_gui: bool

    @classmethod
    def create(
        cls,
        *,
        env_file: Path,
        log_dir: Path | None,
        startup_timeout: float,
        shutdown_timeout: float,
        windowed: bool,
        no_gui: bool,
        inherited_environment: Mapping[str, str] | None = None,
    ) -> LauncherConfig:
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise LauncherConfigurationError("startup timeout must be positive")
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise LauncherConfigurationError("shutdown timeout must be positive")

        resolved_env_file = env_file.expanduser().resolve()
        project_dir = resolved_env_file.parent
        environment = load_environment(resolved_env_file, inherited_environment)
        resolved_log_dir = (
            project_dir / ".signatus" / "logs"
            if log_dir is None
            else _resolve_from(project_dir, log_dir)
        )

        ai_url = environment.get("SIGNATUS_AI_BASE_URL", "http://127.0.0.1:8001")
        core_url = environment.get("SIGNATUS_CORE_URL", "http://127.0.0.1:8000")
        ai_endpoint = parse_local_http_endpoint(ai_url, "SIGNATUS_AI_BASE_URL")
        core_endpoint = parse_local_http_endpoint(core_url, "SIGNATUS_CORE_URL")
        _validate_ai_events_url(environment, ai_endpoint)

        return cls(
            project_dir=project_dir,
            env_file=resolved_env_file,
            log_dir=resolved_log_dir,
            environment=MappingProxyType(environment),
            ai_endpoint=ai_endpoint,
            core_endpoint=core_endpoint,
            startup_timeout=startup_timeout,
            shutdown_timeout=shutdown_timeout,
            windowed=windowed,
            no_gui=no_gui,
        )

    def resolve_runtime_path(self, value: str) -> Path:
        return _resolve_from(self.project_dir, Path(value))


def load_environment(
    env_file: Path,
    inherited_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if not env_file.is_file():
        raise LauncherConfigurationError(f"environment file does not exist: {env_file}")
    try:
        with env_file.open(encoding="utf-8") as stream:
            malformed = next((binding for binding in parse_stream(stream) if binding.error), None)
        if malformed is not None:
            raise LauncherConfigurationError(
                f"cannot parse environment file {env_file} at line {malformed.original.line}"
            )
        values = dotenv_values(env_file)
    except (OSError, UnicodeError) as exc:
        raise LauncherConfigurationError(f"cannot read environment file {env_file}: {exc}") from exc

    file_environment: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            raise LauncherConfigurationError(
                f"environment variable {key!r} has no value in {env_file}"
            )
        file_environment[key] = value

    # Match normal dotenv semantics: explicitly exported process values win.
    inherited = os.environ if inherited_environment is None else inherited_environment
    file_environment.update({key: str(value) for key, value in inherited.items()})
    file_environment["PYTHONUNBUFFERED"] = "1"
    return file_environment


def parse_local_http_endpoint(value: str, variable: str) -> ServiceEndpoint:
    parsed = _parse_url(value, variable)
    if parsed.scheme != "http":
        raise LauncherConfigurationError(f"{variable} must use http")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise LauncherConfigurationError(f"{variable} must be an HTTP base URL without a path")
    if parsed.username is not None or parsed.password is not None:
        raise LauncherConfigurationError(f"{variable} must not contain credentials")
    host = parsed.hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise LauncherConfigurationError(f"{variable} must target the local host")
    try:
        port = 80 if parsed.port is None else parsed.port
    except ValueError as exc:
        raise LauncherConfigurationError(f"{variable} contains an invalid port") from exc
    if port == 0:
        raise LauncherConfigurationError(f"{variable} port must be between 1 and 65535")
    return ServiceEndpoint(host=host, port=port)


def _validate_ai_events_url(
    environment: Mapping[str, str],
    ai_endpoint: ServiceEndpoint,
) -> None:
    value = environment.get("SIGNATUS_AI_EVENTS_URL", "ws://127.0.0.1:8001/ws/events")
    parsed = _parse_url(value, "SIGNATUS_AI_EVENTS_URL")
    if parsed.scheme != "ws" or parsed.path != "/ws/events" or parsed.query or parsed.fragment:
        raise LauncherConfigurationError(
            "SIGNATUS_AI_EVENTS_URL must use the approved ws://.../ws/events interface"
        )
    if parsed.username is not None or parsed.password is not None:
        raise LauncherConfigurationError("SIGNATUS_AI_EVENTS_URL must not contain credentials")
    try:
        port = 80 if parsed.port is None else parsed.port
    except ValueError as exc:
        raise LauncherConfigurationError("SIGNATUS_AI_EVENTS_URL contains an invalid port") from exc
    if port == 0:
        raise LauncherConfigurationError(
            "SIGNATUS_AI_EVENTS_URL port must be between 1 and 65535"
        )
    if parsed.hostname != ai_endpoint.host or port != ai_endpoint.port:
        raise LauncherConfigurationError(
            "SIGNATUS_AI_EVENTS_URL must target the configured local AI Service"
        )


def _parse_url(value: str, variable: str) -> SplitResult:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise LauncherConfigurationError(f"{variable} is not a valid URL") from exc
    if not parsed.scheme or parsed.hostname is None:
        raise LauncherConfigurationError(f"{variable} is not a valid absolute URL")
    return parsed


def _resolve_from(base: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()
