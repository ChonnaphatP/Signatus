from __future__ import annotations

import fcntl
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from .config import LauncherConfig


class SingleInstanceError(RuntimeError):
    """Raised when another launcher owns this deployment's runtime lease."""


class SingleInstanceLock:
    """A process-scoped, non-destructive launcher lease.

    The lock file is deliberately retained after release. Ownership comes from
    the kernel ``flock``, not from the presence or contents of the file, so a
    launcher crash cannot leave a false permanent lock behind.
    """

    def __init__(
        self,
        path: Path,
        *,
        launch_id: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.path = path
        self.launch_id = launch_id
        self._metadata = dict(metadata or {})
        self._descriptor: int | None = None

    @classmethod
    def for_config(cls, config: LauncherConfig, *, launch_id: str) -> SingleInstanceLock:
        # One launcher per operating-system user is intentionally stronger than
        # a project-directory lock: two env files must not race for the same
        # camera, TCP ports, or shared-memory name.
        runtime_dir = Path("/tmp") / f"signatus-{os.getuid()}"
        return cls(
            runtime_dir / "launcher.lock",
            launch_id=launch_id,
            metadata={
                "environment_file": str(config.env_file),
                "ai_endpoint": f"{config.ai_endpoint.host}:{config.ai_endpoint.port}",
                "core_endpoint": f"{config.core_endpoint.host}:{config.core_endpoint.port}",
                "camera_source": config.environment.get("SIGNATUS_CAMERA_SOURCE", "0"),
                "shared_memory_name": config.environment.get(
                    "SIGNATUS_FRAME_SHM_NAME", "signatus_camera_v1"
                ),
            },
        )

    @property
    def acquired(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> None:
        if self._descriptor is not None:
            raise SingleInstanceError("launcher runtime lease was already acquired")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory_status = self.path.parent.lstat()
            if not stat.S_ISDIR(directory_status.st_mode) or directory_status.st_uid != os.geteuid():
                raise OSError("runtime lease directory is not owned by the current user")
            flags = os.O_RDWR | os.O_CREAT
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise SingleInstanceError(
                f"cannot open launcher runtime lease {self.path}: {exc}"
            ) from exc

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = self._read_owner(descriptor)
            os.close(descriptor)
            detail = f" ({owner})" if owner else ""
            raise SingleInstanceError(
                f"another Signatus launcher instance is already running{detail}"
            ) from exc
        except OSError as exc:
            os.close(descriptor)
            raise SingleInstanceError(
                f"cannot acquire launcher runtime lease {self.path}: {exc}"
            ) from exc

        owner = {
            "launch_id": self.launch_id,
            "launcher_pid": os.getpid(),
            **self._metadata,
        }
        try:
            encoded = (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        except OSError as exc:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise SingleInstanceError(
                f"cannot record launcher runtime ownership in {self.path}: {exc}"
            ) from exc
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    @staticmethod
    def _read_owner(descriptor: int) -> str | None:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload = json.loads(os.read(descriptor, 16 * 1024).decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        launch_id = payload.get("launch_id")
        launcher_pid = payload.get("launcher_pid")
        if not isinstance(launch_id, str) or not isinstance(launcher_pid, int):
            return None
        return f"launch_id={launch_id}, pid={launcher_pid}"
