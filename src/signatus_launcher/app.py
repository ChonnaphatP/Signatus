from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from .config import LauncherConfig, LauncherConfigurationError
from .instance import SingleInstanceError, SingleInstanceLock
from .preflight import PreflightReport, run_preflight
from .supervisor import RuntimeLog, Supervisor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight, launch, and supervise the Signatus v1 processes"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file; its directory becomes the child-process working directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="rotating log directory (default: <env directory>/.signatus/logs)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="seconds allowed for each service readiness gate (default: 60)",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=10.0,
        help="seconds allowed for each graceful/terminate/kill shutdown stage (default: 10)",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="launch the GUI in a normal window instead of full-screen kiosk mode",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run static preflight checks without starting any process or opening the camera",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="supervise only the AI Service and Core",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    errors = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    if args.windowed and args.no_gui:
        print("signatus-launch: --windowed cannot be combined with --no-gui", file=errors)
        return 2

    try:
        config = LauncherConfig.create(
            env_file=args.env_file,
            log_dir=args.log_dir,
            startup_timeout=args.startup_timeout,
            shutdown_timeout=args.shutdown_timeout,
            windowed=args.windowed,
            no_gui=args.no_gui,
        )
    except LauncherConfigurationError as exc:
        print(f"signatus-launch: configuration error: {exc}", file=errors)
        print("RESULT: FAIL", file=errors)
        return 2

    if args.check_only:
        report = run_preflight(config, include_gui=not args.no_gui)
        _print_preflight(report, stdout=output, stderr=errors)
        print("No processes were started.", file=output)
        return 0 if report.ok else 1

    launch_id = uuid4().hex
    instance_lock = SingleInstanceLock.for_config(config, launch_id=launch_id)
    try:
        instance_lock.acquire()
    except SingleInstanceError as exc:
        print(f"FATAL: {exc}", file=errors)
        print("RESULT: FAIL", file=errors)
        return 1

    try:
        report = run_preflight(config, include_gui=not args.no_gui)
        _print_preflight(report, stdout=output, stderr=errors)
        if not report.ok:
            return 1

        try:
            runtime_log = RuntimeLog(config.log_dir, console=output, launch_id=launch_id)
        except OSError as exc:
            print(f"signatus-launch: cannot create runtime log: {exc}", file=errors)
            print("RESULT: FAIL", file=errors)
            return 1
        try:
            runtime_log.info("launcher", f"launch ID {launch_id}")
            runtime_log.info("launcher", f"using environment file {config.env_file}")
            runtime_log.info("launcher", f"writing rotating logs to {runtime_log.path}")
            runtime_log.info(
                "launcher",
                f"deployment validation result: {report.outcome.value}",
                validation_severity=("DATA_ERROR" if report.data_errors else None),
            )
            for message in report.data_errors:
                runtime_log.warning(
                    "launcher",
                    message,
                    validation_severity="DATA_ERROR",
                )
            for message in report.warnings:
                runtime_log.warning(
                    "launcher",
                    message,
                    validation_severity="WARNING",
                )
            return Supervisor(config, runtime_log).run()
        finally:
            runtime_log.close()
    finally:
        instance_lock.release()


def _print_preflight(
    report: PreflightReport,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    if report.errors:
        print("Fatal errors:", file=stderr)
        for error in report.errors:
            print(f"  FATAL: {error}", file=stderr)
    if report.data_errors:
        print("Data errors:", file=stdout)
        for error in report.data_errors:
            print(f"  DATA ERROR: {error}", file=stdout)
    if report.warnings:
        print("Warnings:", file=stdout)
        for warning in report.warnings:
            print(f"  WARNING: {warning}", file=stdout)
    destination = stderr if report.errors else stdout
    print(f"RESULT: {report.outcome.value}", file=destination)


if __name__ == "__main__":
    raise SystemExit(main())
