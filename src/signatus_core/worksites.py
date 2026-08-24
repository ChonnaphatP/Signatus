from __future__ import annotations

import json
import math
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from signatus_contracts import (
    SFACE_DESCRIPTOR_DIMENSIONS,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

from .domain import AuthorizedWorker, Worksite
from .ppe import PPE_CLASS_MAP, normalize_class_name


@dataclass(frozen=True, slots=True)
class WorksiteRecord:
    """The validated operational view of one worksite JSON file."""

    source: str
    worksite_id: str | None
    name: str | None
    required_ppe: tuple[str, ...]
    available: bool
    unavailable_reason: str | None
    valid_worker_count: int
    invalid_worker_count: int
    worksite: Worksite | None


@dataclass(frozen=True, slots=True)
class WorksiteCatalog:
    """A stable validation snapshot shared by Core and deployment preflight."""

    records: tuple[WorksiteRecord, ...]
    validation_report: ValidationReport

    @property
    def report(self) -> ValidationReport:
        return self.validation_report

    @property
    def available_worksites(self) -> tuple[Worksite, ...]:
        return tuple(
            record.worksite
            for record in self.records
            if record.available and record.worksite is not None
        )

    @property
    def fatal_issues(self) -> tuple[ValidationIssue, ...]:
        return self._issues_with_severity(ValidationSeverity.FATAL)

    @property
    def data_errors(self) -> tuple[ValidationIssue, ...]:
        return self._issues_with_severity(ValidationSeverity.DATA_ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return self._issues_with_severity(ValidationSeverity.WARNING)

    @property
    def has_fatal_errors(self) -> bool:
        return bool(self.fatal_issues)

    def get(self, worksite_id: str) -> Worksite | None:
        for worksite in self.available_worksites:
            if worksite.worksite_id == worksite_id:
                return worksite
        return None

    def get_record(self, worksite_id: str) -> WorksiteRecord | None:
        for record in self.records:
            if record.worksite_id == worksite_id:
                return record
        return None

    def _issues_with_severity(
        self, severity: ValidationSeverity
    ) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue for issue in self.validation_report.issues if issue.severity is severity
        )


class WorksiteRepository:
    """Load worksite data once and fail closed at the narrowest invalid scope."""

    def __init__(
        self,
        directory: Path,
        *,
        ppe_policy: Mapping[str, object] | None = None,
    ):
        self._directory = directory
        self._ppe_policy = PPE_CLASS_MAP if ppe_policy is None else ppe_policy
        self._catalog: WorksiteCatalog | None = None
        self._catalog_lock = threading.RLock()

    def load_catalog(self) -> WorksiteCatalog:
        with self._catalog_lock:
            if self._catalog is None:
                self._catalog = self._build_catalog()
            return self._catalog

    def refresh_catalog(self) -> WorksiteCatalog:
        """Build and atomically publish a fresh catalog snapshot."""

        catalog = self._build_catalog()
        with self._catalog_lock:
            self._catalog = catalog
        return catalog

    def list_records(self) -> tuple[WorksiteRecord, ...]:
        return self.load_catalog().records

    def list_worksites(self) -> tuple[Worksite, ...]:
        """Return only worksite domain objects that are safe to select."""

        return self.load_catalog().available_worksites

    def get(self, worksite_id: str) -> Worksite | None:
        return self.load_catalog().get(worksite_id)

    def get_record(self, worksite_id: str) -> WorksiteRecord | None:
        return self.load_catalog().get_record(worksite_id)

    @property
    def validation_report(self) -> ValidationReport:
        return self.load_catalog().validation_report

    def _build_catalog(self) -> WorksiteCatalog:
        issues: list[ValidationIssue] = []
        if not self._directory.exists():
            issues.append(
                _issue(
                    ValidationSeverity.FATAL,
                    "WORKSITE_DIRECTORY_MISSING",
                    f"Worksite directory does not exist: {self._directory}",
                    source=str(self._directory),
                )
            )
            return _catalog((), issues)
        if not self._directory.is_dir():
            issues.append(
                _issue(
                    ValidationSeverity.FATAL,
                    "WORKSITE_DIRECTORY_INVALID",
                    f"Worksite configuration path is not a directory: {self._directory}",
                    source=str(self._directory),
                )
            )
            return _catalog((), issues)

        try:
            paths = tuple(sorted(self._directory.glob("*.json")))
        except OSError as error:
            issues.append(
                _issue(
                    ValidationSeverity.FATAL,
                    "WORKSITE_DIRECTORY_UNREADABLE",
                    f"Worksite directory cannot be read: {error}",
                    source=str(self._directory),
                )
            )
            return _catalog((), issues)

        if not paths:
            issues.append(
                _issue(
                    ValidationSeverity.WARNING,
                    "NO_WORKSITE_CONFIGURATIONS",
                    f"Worksite directory contains no JSON configurations: {self._directory}",
                    source=str(self._directory),
                )
            )
            return _catalog((), issues)

        records: list[WorksiteRecord] = []
        for path in paths:
            record, file_issues = self._load_record(path)
            records.append(record)
            issues.extend(file_issues)

        duplicate_ids = {
            worksite_id
            for worksite_id, count in Counter(
                record.worksite_id
                for record in records
                if record.worksite_id is not None
            ).items()
            if count > 1
        }
        if duplicate_ids:
            updated_records: list[WorksiteRecord] = []
            for record in records:
                if record.worksite_id not in duplicate_ids:
                    updated_records.append(record)
                    continue
                message = f"Duplicate Wo.No. {record.worksite_id!r}; all duplicates are unavailable."
                issues.append(
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        "DUPLICATE_WORKSITE_ID",
                        message,
                        source=record.source,
                        worksite_id=record.worksite_id,
                    )
                )
                updated_records.append(
                    replace(
                        record,
                        available=False,
                        unavailable_reason=record.unavailable_reason or message,
                        worksite=None,
                    )
                )
            records = updated_records

        # A catalog with no usable worksite is an intentional safe standby state.
        # Only unreadable deployment infrastructure is fatal.
        return _catalog(tuple(records), issues)

    def _load_record(self, path: Path) -> tuple[WorksiteRecord, list[ValidationIssue]]:
        source = path.name
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            message = f"Wo.No. file cannot be read as JSON: {error}"
            return (
                WorksiteRecord(
                    source=source,
                    worksite_id=None,
                    name=None,
                    required_ppe=(),
                    available=False,
                    unavailable_reason=message,
                    valid_worker_count=0,
                    invalid_worker_count=0,
                    worksite=None,
                ),
                [
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        "INVALID_WORKSITE_JSON",
                        message,
                        source=source,
                    )
                ],
            )

        if not isinstance(raw, dict):
            message = "Wo.No. record must be a JSON object."
            return (
                WorksiteRecord(
                    source=source,
                    worksite_id=None,
                    name=None,
                    required_ppe=(),
                    available=False,
                    unavailable_reason=message,
                    valid_worker_count=0,
                    invalid_worker_count=0,
                    worksite=None,
                ),
                [
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        "INVALID_WORKSITE_RECORD",
                        message,
                        source=source,
                    )
                ],
            )

        return self.validate_payload(raw, source=source, reject_example=True)

    def validate_payload(
        self,
        raw: object,
        *,
        source: str,
        reject_example: bool = False,
    ) -> tuple[WorksiteRecord, list[ValidationIssue]]:
        """Validate one decoded compact Wo.No. document without changing the catalog."""

        if not isinstance(raw, dict):
            message = "Wo.No. record must be a JSON object."
            return (
                WorksiteRecord(
                    source=source,
                    worksite_id=None,
                    name=None,
                    required_ppe=(),
                    available=False,
                    unavailable_reason=message,
                    valid_worker_count=0,
                    invalid_worker_count=0,
                    worksite=None,
                ),
                [
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        "INVALID_WORKSITE_RECORD",
                        message,
                        source=source,
                    )
                ],
            )

        issues: list[ValidationIssue] = []
        worksite_errors: list[str] = []
        worksite_id = _nonempty_string(raw.get("worksite_id"))

        if reject_example and ".example." in source:
            message = "Example Wo.No. data is not available for operational selection."
            worksite_errors.append(message)
            issues.append(
                _issue(
                    ValidationSeverity.DATA_ERROR,
                    "EXAMPLE_WORKSITE_DATA",
                    message,
                    source=source,
                    worksite_id=worksite_id,
                )
            )

        if worksite_id is None:
            message = "Wo.No. record is missing a non-empty string worksite_id."
            worksite_errors.append(message)
            issues.append(
                _issue(
                    ValidationSeverity.DATA_ERROR,
                    "INVALID_WORKSITE_ID",
                    message,
                    source=source,
                )
            )

        name = _nonempty_string(raw.get("name"))
        if name is None:
            message = "Wo.No. record is missing a non-empty string name."
            worksite_errors.append(message)
            issues.append(
                _issue(
                    ValidationSeverity.DATA_ERROR,
                    "INVALID_WORKSITE_NAME",
                    message,
                    source=source,
                    worksite_id=worksite_id,
                )
            )

        required_ppe, ppe_messages = self._validate_required_ppe(raw.get("required_ppe"))
        for code, message in ppe_messages:
            worksite_errors.append(message)
            issues.append(
                _issue(
                    ValidationSeverity.DATA_ERROR,
                    code,
                    message,
                    source=source,
                    worksite_id=worksite_id,
                )
            )

        raw_workers = raw.get("authorized_workers")
        workers: tuple[AuthorizedWorker, ...] = ()
        invalid_worker_count = 0
        if not isinstance(raw_workers, list):
            message = "Wo.No. record is missing an authorized_workers list."
            worksite_errors.append(message)
            issues.append(
                _issue(
                    ValidationSeverity.DATA_ERROR,
                    "INVALID_AUTHORIZED_WORKERS",
                    message,
                    source=source,
                    worksite_id=worksite_id,
                )
            )
        else:
            workers, invalid_worker_count, worker_issues = _validate_workers(
                raw_workers,
                source=source,
                worksite_id=worksite_id,
            )
            issues.extend(worker_issues)
            if not workers:
                message = "Wo.No. contains no usable enrolled workers."
                worksite_errors.append(message)
                issues.append(
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        "NO_USABLE_AUTHORIZED_WORKERS",
                        message,
                        source=source,
                        worksite_id=worksite_id,
                    )
                )

        available = not worksite_errors
        worksite = (
            Worksite(
                worksite_id=worksite_id,
                name=name,
                authorized_workers=workers,
                required_ppe=required_ppe,
            )
            if available and worksite_id is not None and name is not None
            else None
        )
        return (
            WorksiteRecord(
                source=source,
                worksite_id=worksite_id,
                name=name,
                required_ppe=required_ppe,
                available=available,
                unavailable_reason=None if available else worksite_errors[0],
                valid_worker_count=len(workers),
                invalid_worker_count=invalid_worker_count,
                worksite=worksite,
            ),
            issues,
        )

    def _validate_required_ppe(
        self, value: object
    ) -> tuple[tuple[str, ...], list[tuple[str, str]]]:
        if not isinstance(value, list):
            return (), [
                (
                    "INVALID_REQUIRED_PPE",
                    "Wo.No. record is missing a required_ppe list.",
                )
            ]

        items: list[str] = []
        messages: list[tuple[str, str]] = []
        normalized_items: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                messages.append(
                    (
                        "INVALID_REQUIRED_PPE_ITEM",
                        "Required PPE entries must be non-empty strings.",
                    )
                )
                continue
            normalized = normalize_class_name(item)
            if normalized in normalized_items:
                messages.append(
                    (
                        "DUPLICATE_REQUIRED_PPE",
                        f"Required PPE item {item!r} is duplicated.",
                    )
                )
            normalized_items.add(normalized)
            if normalized not in self._ppe_policy:
                messages.append(
                    (
                        "UNKNOWN_PPE_POLICY",
                        f"Required PPE item {item!r} has no approved Core policy.",
                    )
                )
            items.append(item)
        return tuple(items), messages


def _validate_workers(
    entries: list[object],
    *,
    source: str,
    worksite_id: str | None,
) -> tuple[tuple[AuthorizedWorker, ...], int, list[ValidationIssue]]:
    worker_ids = [
        worker_id
        for entry in entries
        if isinstance(entry, dict)
        if (worker_id := _nonempty_string(entry.get("worker_id"))) is not None
    ]
    duplicate_ids = {
        worker_id for worker_id, count in Counter(worker_ids).items() if count > 1
    }

    workers: list[AuthorizedWorker] = []
    issues: list[ValidationIssue] = []
    invalid_count = 0
    for index, entry in enumerate(entries):
        entry_issues: list[ValidationIssue] = []
        if not isinstance(entry, dict):
            entry_issues.append(
                _issue(
                    ValidationSeverity.DATA_ERROR,
                    "INVALID_WORKER_RECORD",
                    f"Worker record {index + 1} must be a JSON object.",
                    source=source,
                    worksite_id=worksite_id,
                )
            )
            worker_id = None
            embedding = None
        else:
            worker_id = _nonempty_string(entry.get("worker_id"))
            if worker_id is None:
                entry_issues.append(
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        "INVALID_WORKER_ID",
                        f"Worker record {index + 1} is missing a non-empty string worker_id.",
                        source=source,
                        worksite_id=worksite_id,
                    )
                )
            elif worker_id in duplicate_ids:
                entry_issues.append(
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        "DUPLICATE_WORKER_ID",
                        f"Worker ID {worker_id!r} is duplicated; all duplicates are unavailable.",
                        source=source,
                        worksite_id=worksite_id,
                        worker_id=worker_id,
                    )
                )
            worker_name = _nonempty_string(entry.get("name"))
            if worker_name is None:
                entry_issues.append(
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        "INVALID_WORKER_NAME",
                        f"Worker record {index + 1} is missing a non-empty string name.",
                        source=source,
                        worksite_id=worksite_id,
                        worker_id=worker_id,
                    )
                )
            embedding, embedding_issue = _validate_embedding(entry.get("embedding"))
            if embedding_issue is not None:
                code, message = embedding_issue
                entry_issues.append(
                    _issue(
                        ValidationSeverity.DATA_ERROR,
                        code,
                        message,
                        source=source,
                        worksite_id=worksite_id,
                        worker_id=worker_id,
                    )
                )

        if entry_issues:
            invalid_count += 1
            issues.extend(entry_issues)
            continue
        if worker_id is not None and worker_name is not None and embedding is not None:
            workers.append(
                AuthorizedWorker(worker_id=worker_id, name=worker_name, embedding=embedding)
            )

    return tuple(workers), invalid_count, issues


def _validate_embedding(
    value: object,
) -> tuple[tuple[float, ...] | None, tuple[str, str] | None]:
    if value is None:
        return None, (
            "MISSING_SFACE_DESCRIPTOR",
            f"Missing SFace descriptor; expected {SFACE_DESCRIPTOR_DIMENSIONS} values.",
        )
    if not isinstance(value, list):
        return None, (
            "INVALID_SFACE_DESCRIPTOR_TYPE",
            "Invalid SFace descriptor type; expected a JSON list of numeric values.",
        )
    if len(value) != SFACE_DESCRIPTOR_DIMENSIONS:
        return None, (
            "INVALID_SFACE_DESCRIPTOR_LENGTH",
            (
                f"Invalid SFace descriptor. Expected {SFACE_DESCRIPTOR_DIMENSIONS} values, "
                f"found {len(value)}."
            ),
        )
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        return None, (
            "INVALID_SFACE_DESCRIPTOR_TYPE",
            "Invalid SFace descriptor type; all 128 values must be numeric.",
        )
    try:
        embedding = tuple(float(item) for item in value)
    except (OverflowError, ValueError):
        return None, (
            "INVALID_SFACE_DESCRIPTOR_VALUE",
            "Invalid SFace descriptor; its numeric values are unusable.",
        )
    if any(not math.isfinite(item) for item in embedding):
        return None, (
            "NONFINITE_SFACE_DESCRIPTOR",
            "Invalid SFace descriptor; values must not contain NaN or infinity.",
        )
    norm = math.hypot(*embedding)
    if norm == 0.0 or not math.isfinite(norm):
        return None, (
            "UNUSABLE_SFACE_DESCRIPTOR",
            "Invalid SFace descriptor; the descriptor must have a finite nonzero norm.",
        )
    return embedding, None


def _nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _issue(
    severity: ValidationSeverity,
    code: str,
    message: str,
    *,
    source: str | None = None,
    worksite_id: str | None = None,
    worker_id: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        source=source,
        worksite_id=worksite_id,
        worker_id=worker_id,
    )


def _catalog(
    records: tuple[WorksiteRecord, ...], issues: list[ValidationIssue]
) -> WorksiteCatalog:
    return WorksiteCatalog(
        records=records,
        validation_report=ValidationReport(issues=issues),
    )
