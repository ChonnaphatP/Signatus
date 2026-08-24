from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx

from signatus_contracts import ValidationIssue, ValidationSeverity

from .domain import AuthorizedWorker, Worksite
from .ppe import PPE_CLASS_MAP
from .worker_profiles import validate_worker_profile as validate_stored_worker_profile
from .worksites import WorksiteCatalog, WorksiteRecord, WorksiteRepository

DEFAULT_URL_TIMEOUT_SECONDS: Final = 10.0
DEFAULT_URL_MAX_BYTES: Final = 2_000_000
_SAFE_FILENAME_RUN = re.compile(r"[^A-Za-z0-9._-]+")
_REPEATED_UNDERSCORES = re.compile(r"_+")
_WORKER_ISSUE_CODES: Final = frozenset(
    {
        "INVALID_WORKER_RECORD",
        "INVALID_WORKER_ID",
        "INVALID_WORKER_NAME",
        "DUPLICATE_WORKER_ID",
        "MISSING_SFACE_DESCRIPTOR",
        "INVALID_SFACE_DESCRIPTOR_TYPE",
        "INVALID_SFACE_DESCRIPTOR_LENGTH",
        "INVALID_SFACE_DESCRIPTOR_VALUE",
        "NONFINITE_SFACE_DESCRIPTOR",
        "UNUSABLE_SFACE_DESCRIPTOR",
    }
)


@dataclass(frozen=True, slots=True)
class IssueSummary:
    severity: str
    code: str
    message: str
    source: str | None = None
    worksite_id: str | None = None
    worker_id: str | None = None

    @classmethod
    def from_validation_issue(cls, issue: ValidationIssue) -> IssueSummary:
        return cls(
            severity=issue.severity.value,
            code=issue.code,
            message=issue.message,
            source=issue.source,
            worksite_id=issue.worksite_id,
            worker_id=issue.worker_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "worksite_id": self.worksite_id,
            "worker_id": self.worker_id,
        }


@dataclass(frozen=True, slots=True)
class WorkerSummary:
    worker_id: str
    name: str
    embedding: tuple[float, ...]

    @classmethod
    def from_worker(cls, worker: AuthorizedWorker) -> WorkerSummary:
        return cls(
            worker_id=worker.worker_id,
            name=worker.name,
            embedding=worker.embedding,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "embedding": list(self.embedding),
        }


@dataclass(frozen=True, slots=True)
class ManagedWorksiteEntry:
    source: str
    worksite_id: str | None
    name: str | None
    available: bool
    unavailable_reason: str | None
    valid_worker_count: int
    invalid_worker_count: int
    required_ppe: tuple[str, ...]
    workers: tuple[WorkerSummary, ...]
    issues: tuple[IssueSummary, ...]
    invalid_worker_issues: tuple[IssueSummary, ...]
    raw_json: str
    parse_error: str | None

    @property
    def validity_state(self) -> str:
        if not self.available:
            return "INVALID"
        if self.invalid_worker_count:
            return "PARTIAL"
        return "VALID"

    def to_dict(
        self,
        active_worksite_id: str | None = None,
        active_source: str | None = None,
    ) -> dict[str, object]:
        is_active = (
            active_source is not None
            and active_source == self.source
            or active_worksite_id is not None
            and active_worksite_id == self.worksite_id
        )
        return {
            "source": self.source,
            "worksite_id": self.worksite_id,
            "name": self.name,
            "available": self.available,
            "validity_state": self.validity_state,
            "unavailable_reason": self.unavailable_reason,
            "worker_count": self.valid_worker_count,
            "valid_worker_count": self.valid_worker_count,
            "invalid_worker_count": self.invalid_worker_count,
            "required_ppe": list(self.required_ppe),
            "authorized_workers": [worker.to_dict() for worker in self.workers],
            "issues": [issue.to_dict() for issue in self.issues],
            "invalid_worker_issues": [issue.to_dict() for issue in self.invalid_worker_issues],
            "raw_json": self.raw_json,
            "parse_error": self.parse_error,
            "active": is_active,
        }


@dataclass(frozen=True, slots=True)
class JsonView:
    source: str
    raw: str
    formatted: str | None
    parse_error: str | None
    read_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "raw": self.raw,
            "formatted": self.formatted,
            "parse_error": self.parse_error,
            "read_only": self.read_only,
        }


@dataclass(frozen=True, slots=True)
class ImportDocument:
    source_name: str
    content: str | bytes


@dataclass(frozen=True, slots=True)
class URLFetchResponse:
    status_code: int
    content: str | bytes
    content_type: str | None = None


@dataclass(frozen=True, slots=True)
class OperationResult:
    status: str
    source: str
    worksite_id: str | None
    message: str
    destination: str | None = None
    issues: tuple[IssueSummary, ...] = ()
    skipped_workers: tuple[IssueSummary, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status in {"CREATED", "UPDATED", "IMPORTED", "DELETED"}

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source": self.source,
            "worksite_id": self.worksite_id,
            "message": self.message,
            "destination": self.destination,
            "issues": [issue.to_dict() for issue in self.issues],
            "skipped_workers": [issue.to_dict() for issue in self.skipped_workers],
        }


@dataclass(frozen=True, slots=True)
class BatchImportResult:
    results: tuple[OperationResult, ...]

    @property
    def imported(self) -> tuple[OperationResult, ...]:
        return tuple(result for result in self.results if result.status == "IMPORTED")

    @property
    def skipped(self) -> tuple[OperationResult, ...]:
        return tuple(result for result in self.results if result.status == "SKIPPED")

    @property
    def failed(self) -> tuple[OperationResult, ...]:
        return tuple(result for result in self.results if result.status == "FAILED")

    def to_dict(self) -> dict[str, object]:
        return {
            "imported": [result.to_dict() for result in self.imported],
            "skipped": [result.to_dict() for result in self.skipped],
            "failed": [result.to_dict() for result in self.failed],
            "imported_count": len(self.imported),
            "skipped_count": len(self.skipped),
            "failed_count": len(self.failed),
        }


class WorksiteManagementError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        issues: Sequence[IssueSummary] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.issues = tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "issues": [issue.to_dict() for issue in self.issues],
        }


URLFetcher = Callable[[str, float, int], Awaitable[URLFetchResponse | str | bytes]]


class WorksiteManagementService:
    """Maintain deployment Wo.No. files without mutating Core runtime state."""

    def __init__(
        self,
        directory: Path,
        repository: WorksiteRepository | None = None,
        *,
        fetcher: URLFetcher | None = None,
        url_timeout_seconds: float = DEFAULT_URL_TIMEOUT_SECONDS,
        url_max_bytes: int = DEFAULT_URL_MAX_BYTES,
    ) -> None:
        if url_timeout_seconds <= 0:
            raise ValueError("URL timeout must be positive")
        if url_max_bytes <= 0:
            raise ValueError("URL response-size limit must be positive")
        self._directory = directory
        self._repository = repository or WorksiteRepository(directory)
        self._fetcher = fetcher or self._fetch_with_httpx
        self._url_timeout_seconds = url_timeout_seconds
        self._url_max_bytes = url_max_bytes
        self._mutation_lock = threading.RLock()

    @property
    def ppe_options(self) -> tuple[str, ...]:
        """Return the semantic PPE choices owned by Core's deployed policy."""

        return tuple(PPE_CLASS_MAP)

    def list_entries(self, *, refresh: bool = False) -> tuple[ManagedWorksiteEntry, ...]:
        with self._mutation_lock:
            catalog = (
                self._repository.refresh_catalog() if refresh else self._repository.load_catalog()
            )
            return self._entries_from_catalog(catalog)

    def refresh(self) -> tuple[ManagedWorksiteEntry, ...]:
        return self.list_entries(refresh=True)

    def details(self, source: str) -> ManagedWorksiteEntry:
        safe_source = self._safe_source_name(source)
        for entry in self.list_entries():
            if entry.source == safe_source:
                return entry
        self._resolve_source(safe_source)
        raise WorksiteManagementError(
            "WORKSITE_NOT_IN_CATALOG",
            f"Wo.No. file {safe_source!r} is not present in the current catalog; refresh first.",
        )

    def view_json(self, source: str) -> JsonView:
        path = self._resolve_source(source)
        raw, decode_error = _read_text(path)
        if decode_error is not None:
            return JsonView(
                source=path.name,
                raw=raw,
                formatted=None,
                parse_error=decode_error,
            )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            return JsonView(
                source=path.name,
                raw=raw,
                formatted=None,
                parse_error=str(error),
            )
        try:
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            return JsonView(
                source=path.name,
                raw=raw,
                formatted=None,
                parse_error=f"JSON contains an unsupported value: {error}",
            )
        return JsonView(
            source=path.name,
            raw=raw,
            formatted=formatted,
            parse_error=None,
        )

    def validate_worker_profile(self, raw: object) -> dict[str, object]:
        result = validate_stored_worker_profile(raw, source="worker-profile")
        if not result.valid or result.profile is None:
            raise WorksiteManagementError(
                "INVALID_WORKER_PROFILE",
                result.errors[0].message if result.errors else "Worker Profile is invalid.",
            )
        return result.profile.to_dict()

    def worker_from_profile(self, raw: object) -> dict[str, object]:
        return self.validate_worker_profile(raw)

    def create(self, payload: object) -> OperationResult:
        with self._mutation_lock:
            compact = self._validate_complete_payload(payload, source="new-worksite")
            worksite_id = str(compact["worksite_id"])
            self._assert_unique_worksite_id(worksite_id)
            source = sanitize_worksite_filename(worksite_id)
            destination = self._destination_for_new_source(source)
            self._write_json(destination, compact, overwrite=False)
            self._repository.refresh_catalog()
            return OperationResult(
                status="CREATED",
                source=destination.name,
                worksite_id=worksite_id,
                destination=destination.name,
                message=f"Created Wo.No. {worksite_id}.",
            )

    def edit(self, source: str, payload: object) -> OperationResult:
        with self._mutation_lock:
            path = self._resolve_source(source)
            existing_id = self._read_internal_worksite_id(path)
            compact = self._validate_complete_payload(payload, source=path.name)
            edited_id = str(compact["worksite_id"])
            if edited_id != existing_id:
                raise WorksiteManagementError(
                    "IMMUTABLE_WORKSITE_ID",
                    "The Worksite ID cannot be changed while editing a Wo.No.",
                )
            self._assert_unique_worksite_id(edited_id, exclude_source=path.name)
            self._write_json(path, compact, overwrite=True)
            self._repository.refresh_catalog()
            return OperationResult(
                status="UPDATED",
                source=path.name,
                worksite_id=edited_id,
                destination=path.name,
                message=f"Updated Wo.No. {edited_id} on disk.",
            )

    def import_documents(
        self, documents: Sequence[ImportDocument | Mapping[str, object]]
    ) -> BatchImportResult:
        results: list[OperationResult] = []
        mutated = False
        with self._mutation_lock:
            for document in documents:
                try:
                    normalized = _normalize_import_document(document)
                except WorksiteManagementError as error:
                    source_name = (
                        document.get("source_name", "<import>")
                        if isinstance(document, Mapping)
                        else getattr(document, "source_name", "<import>")
                    )
                    results.append(
                        OperationResult(
                            status="FAILED",
                            source=(
                                source_name
                                if isinstance(source_name, str)
                                else "<import>"
                            ),
                            worksite_id=None,
                            message=error.message,
                            issues=error.issues,
                        )
                    )
                    continue
                result = self._import_document(normalized)
                results.append(result)
                mutated = mutated or result.status == "IMPORTED"
            if mutated:
                self._repository.refresh_catalog()
        return BatchImportResult(tuple(results))

    async def import_url(self, url: str) -> OperationResult:
        parsed_url = urlsplit(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise WorksiteManagementError(
                "INVALID_IMPORT_URL",
                "Import URL must be one direct http:// or https:// JSON URL.",
            )
        if parsed_url.username is not None or parsed_url.password is not None:
            raise WorksiteManagementError(
                "URL_AUTHENTICATION_UNSUPPORTED",
                "Authenticated URL imports are not supported.",
            )

        try:
            fetched = await asyncio.wait_for(
                self._fetcher(
                    url,
                    self._url_timeout_seconds,
                    self._url_max_bytes,
                ),
                timeout=self._url_timeout_seconds,
            )
        except TimeoutError as error:
            raise WorksiteManagementError(
                "URL_IMPORT_TIMEOUT",
                "The direct JSON URL did not respond before the timeout.",
            ) from error
        except WorksiteManagementError:
            raise
        except httpx.HTTPStatusError as error:
            raise WorksiteManagementError(
                "URL_HTTP_ERROR",
                f"The direct JSON URL returned HTTP {error.response.status_code}.",
            ) from error
        except httpx.HTTPError as error:
            raise WorksiteManagementError(
                "URL_FETCH_FAILED",
                f"The direct JSON URL could not be retrieved: {error}",
            ) from error
        except Exception as error:
            raise WorksiteManagementError(
                "URL_FETCH_FAILED",
                f"The direct JSON URL could not be retrieved: {error}",
            ) from error

        response = (
            fetched
            if isinstance(fetched, URLFetchResponse)
            else URLFetchResponse(status_code=200, content=fetched)
        )
        if not 200 <= response.status_code < 300:
            raise WorksiteManagementError(
                "URL_HTTP_ERROR",
                f"The direct JSON URL returned HTTP {response.status_code}.",
            )
        content_bytes = (
            response.content.encode("utf-8")
            if isinstance(response.content, str)
            else response.content
        )
        if not isinstance(content_bytes, bytes):
            raise WorksiteManagementError(
                "INVALID_URL_CONTENT",
                "The direct JSON URL response was not text or bytes.",
            )
        if not content_bytes:
            raise WorksiteManagementError(
                "EMPTY_URL_CONTENT",
                "The direct JSON URL returned an empty response.",
            )
        if len(content_bytes) > self._url_max_bytes:
            raise WorksiteManagementError(
                "URL_RESPONSE_TOO_LARGE",
                f"The direct JSON URL exceeded {self._url_max_bytes} bytes.",
            )
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WorksiteManagementError(
                "INVALID_URL_CONTENT",
                "The direct JSON URL response is not valid UTF-8 JSON.",
            ) from error

        batch = self.import_documents((ImportDocument(source_name=url, content=content),))
        return batch.results[0]

    def delete(
        self,
        source: str,
        *,
        confirmed: bool,
        active_worksite_id: str | None = None,
        active_source: str | None = None,
    ) -> OperationResult:
        if not confirmed:
            raise WorksiteManagementError(
                "DELETE_CONFIRMATION_REQUIRED",
                "Deleting a Wo.No. requires explicit operator confirmation.",
            )
        with self._mutation_lock:
            path = self._resolve_source(source)
            worksite_id = self._try_read_internal_worksite_id(path)
            if active_source == path.name or (
                active_worksite_id is not None and active_worksite_id == worksite_id
            ):
                raise WorksiteManagementError(
                    "ACTIVE_WORKSITE_DELETE_FORBIDDEN",
                    "The active Wo.No. cannot be deleted. Select another Wo.No. first.",
                )
            try:
                path.unlink()
            except OSError as error:
                raise WorksiteManagementError(
                    "WORKSITE_DELETE_FAILED",
                    f"Wo.No. file {path.name!r} could not be deleted: {error}",
                ) from error
            self._repository.refresh_catalog()
            return OperationResult(
                status="DELETED",
                source=path.name,
                worksite_id=worksite_id,
                message=f"Deleted Wo.No. file {path.name}.",
            )

    def _entries_from_catalog(self, catalog: WorksiteCatalog) -> tuple[ManagedWorksiteEntry, ...]:
        issues_by_source: dict[str, list[IssueSummary]] = {}
        for issue in catalog.validation_report.issues:
            if issue.source is not None:
                issues_by_source.setdefault(issue.source, []).append(
                    IssueSummary.from_validation_issue(issue)
                )
        return tuple(
            self._entry_from_record(record, tuple(issues_by_source.get(record.source, ())))
            for record in catalog.records
        )

    def _entry_from_record(
        self,
        record: WorksiteRecord,
        issues: tuple[IssueSummary, ...],
    ) -> ManagedWorksiteEntry:
        try:
            path = self._resolve_source(record.source)
            raw, decode_error = _read_text(path)
        except WorksiteManagementError as error:
            raw = ""
            decode_error = error.message

        parsed: object | None = None
        parse_error = decode_error
        if parse_error is None:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as error:
                parse_error = str(error)

        workers: tuple[WorkerSummary, ...] = ()
        if isinstance(parsed, dict):
            local_record, _ = self._repository.validate_payload(
                parsed,
                source=record.source,
                reject_example=True,
            )
            if local_record.worksite is not None:
                workers = tuple(
                    WorkerSummary.from_worker(worker)
                    for worker in local_record.worksite.authorized_workers
                )
        invalid_worker_issues = tuple(
            issue
            for issue in issues
            if issue.worker_id is not None or issue.code in _WORKER_ISSUE_CODES
        )
        return ManagedWorksiteEntry(
            source=record.source,
            worksite_id=record.worksite_id,
            name=record.name,
            available=record.available,
            unavailable_reason=record.unavailable_reason,
            valid_worker_count=record.valid_worker_count,
            invalid_worker_count=record.invalid_worker_count,
            required_ppe=record.required_ppe,
            workers=workers,
            issues=issues,
            invalid_worker_issues=invalid_worker_issues,
            raw_json=raw,
            parse_error=parse_error,
        )

    def _validate_complete_payload(self, raw: object, *, source: str) -> dict[str, object]:
        compact = _compact_payload(raw)
        record, issues = self._repository.validate_payload(compact, source=source)
        summaries = _issue_summaries(issues)
        if not record.available or record.worksite is None or record.invalid_worker_count:
            raise WorksiteManagementError(
                "INVALID_WORKSITE",
                _first_issue_message(summaries, "Wo.No. configuration is invalid."),
                issues=summaries,
            )
        return _payload_from_worksite(record.worksite)

    def _import_document(self, document: ImportDocument) -> OperationResult:
        try:
            text = _document_text(document.content)
            raw = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            issue = IssueSummary(
                severity=ValidationSeverity.DATA_ERROR.value,
                code="INVALID_WORKSITE_JSON",
                message=f"Wo.No. import is not valid UTF-8 JSON: {error}",
                source=document.source_name,
            )
            return OperationResult(
                status="FAILED",
                source=document.source_name,
                worksite_id=None,
                message=issue.message,
                issues=(issue,),
            )

        compact = _compact_payload(raw)
        record, issues = self._repository.validate_payload(
            compact,
            source=document.source_name,
        )
        summaries = _issue_summaries(issues)
        if not record.available or record.worksite is None:
            return OperationResult(
                status="FAILED",
                source=document.source_name,
                worksite_id=record.worksite_id,
                message=_first_issue_message(summaries, "Wo.No. import validation failed."),
                issues=summaries,
            )

        worksite = record.worksite
        skipped_workers = tuple(
            issue
            for issue in summaries
            if issue.worker_id is not None or issue.code in _WORKER_ISSUE_CODES
        )
        duplicate_source = self._find_worksite_id_source(worksite.worksite_id)
        if duplicate_source is not None:
            return OperationResult(
                status="SKIPPED",
                source=document.source_name,
                worksite_id=worksite.worksite_id,
                message=(f"Wo.No. {worksite.worksite_id!r} already exists in {duplicate_source}."),
                issues=summaries,
                skipped_workers=skipped_workers,
            )

        destination_name = sanitize_worksite_filename(worksite.worksite_id)
        try:
            destination = self._destination_for_new_source(destination_name)
            self._write_json(
                destination,
                _payload_from_worksite(worksite),
                overwrite=False,
            )
        except WorksiteManagementError as error:
            return OperationResult(
                status="FAILED",
                source=document.source_name,
                worksite_id=worksite.worksite_id,
                message=error.message,
                issues=(*summaries, *error.issues),
                skipped_workers=skipped_workers,
            )

        skipped_suffix = (
            f" Skipped {record.invalid_worker_count} invalid worker record(s)."
            if record.invalid_worker_count
            else ""
        )
        return OperationResult(
            status="IMPORTED",
            source=document.source_name,
            worksite_id=worksite.worksite_id,
            destination=destination.name,
            message=f"Imported Wo.No. {worksite.worksite_id}.{skipped_suffix}",
            issues=summaries,
            skipped_workers=skipped_workers,
        )

    def _assert_unique_worksite_id(
        self,
        worksite_id: str,
        *,
        exclude_source: str | None = None,
    ) -> None:
        source = self._find_worksite_id_source(worksite_id, exclude_source=exclude_source)
        if source is not None:
            raise WorksiteManagementError(
                "DUPLICATE_WORKSITE_ID",
                f"Wo.No. {worksite_id!r} already exists in {source}.",
            )

    def _find_worksite_id_source(
        self,
        worksite_id: str,
        *,
        exclude_source: str | None = None,
    ) -> str | None:
        self._ensure_directory()
        try:
            paths = sorted(self._directory.glob("*.json"))
        except OSError as error:
            raise WorksiteManagementError(
                "WORKSITE_DIRECTORY_UNREADABLE",
                f"Worksite directory cannot be scanned: {error}",
            ) from error
        for path in paths:
            if path.name == exclude_source:
                continue
            if self._try_read_internal_worksite_id(path) == worksite_id:
                return path.name
        return None

    def _read_internal_worksite_id(self, path: Path) -> str:
        worksite_id = self._try_read_internal_worksite_id(path)
        if worksite_id is None:
            raise WorksiteManagementError(
                "INVALID_EXISTING_WORKSITE",
                f"Wo.No. file {path.name!r} has no readable internal worksite_id.",
            )
        return worksite_id

    @staticmethod
    def _try_read_internal_worksite_id(path: Path) -> str | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        value = raw.get("worksite_id")
        return value if isinstance(value, str) and value.strip() else None

    def _destination_for_new_source(self, source: str) -> Path:
        self._ensure_directory()
        safe_source = self._safe_source_name(source)
        destination = self._directory / safe_source
        if destination.exists() or destination.is_symlink():
            raise WorksiteManagementError(
                "WORKSITE_FILENAME_COLLISION",
                f"Wo.No. destination {safe_source!r} already exists and was not overwritten.",
            )
        return destination

    def _resolve_source(self, source: str) -> Path:
        self._ensure_directory()
        safe_source = self._safe_source_name(source)
        candidate = self._directory / safe_source
        if candidate.is_symlink():
            raise WorksiteManagementError(
                "UNSAFE_WORKSITE_SOURCE",
                "Symbolic links are not valid Wo.No. sources.",
            )
        try:
            root = self._directory.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise WorksiteManagementError(
                "WORKSITE_NOT_FOUND",
                f"Wo.No. file {safe_source!r} was not found.",
            ) from error
        except OSError as error:
            raise WorksiteManagementError(
                "WORKSITE_SOURCE_UNREADABLE",
                f"Wo.No. file {safe_source!r} cannot be resolved: {error}",
            ) from error
        if resolved.parent != root or not resolved.is_file():
            raise WorksiteManagementError(
                "UNSAFE_WORKSITE_SOURCE",
                "Wo.No. source must be a JSON file directly under config/worksites/.",
            )
        return resolved

    @staticmethod
    def _safe_source_name(source: str) -> str:
        if (
            not isinstance(source, str)
            or not source
            or source != Path(source).name
            or Path(source).suffix.casefold() != ".json"
            or source in {".", ".."}
            or "/" in source
            or "\\" in source
            or "\x00" in source
        ):
            raise WorksiteManagementError(
                "UNSAFE_WORKSITE_SOURCE",
                "Wo.No. source must be a JSON basename without directory components.",
            )
        return source

    def _ensure_directory(self) -> None:
        if not self._directory.exists() or not self._directory.is_dir():
            raise WorksiteManagementError(
                "WORKSITE_DIRECTORY_UNAVAILABLE",
                f"Worksite directory is unavailable: {self._directory}",
            )

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object], *, overwrite: bool) -> None:
        try:
            serialized = (
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        except (TypeError, ValueError) as error:
            raise WorksiteManagementError(
                "WORKSITE_SERIALIZATION_FAILED",
                f"Wo.No. JSON could not be serialized: {error}",
            ) from error
        _atomic_write_text(path, serialized, overwrite=overwrite)

    async def _fetch_with_httpx(
        self,
        url: str,
        timeout_seconds: float,
        max_bytes: int,
    ) -> URLFetchResponse:
        chunks: list[bytes] = []
        received = 0
        async with (
            httpx.AsyncClient(
                timeout=timeout_seconds,
                follow_redirects=True,
            ) as client,
            client.stream(
                "GET",
                url,
                headers={"Accept": "application/json"},
            ) as response,
        ):
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        raise WorksiteManagementError(
                            "URL_RESPONSE_TOO_LARGE",
                            f"The direct JSON URL exceeded {max_bytes} bytes.",
                        )
                except ValueError:
                    pass
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > max_bytes:
                    raise WorksiteManagementError(
                        "URL_RESPONSE_TOO_LARGE",
                        f"The direct JSON URL exceeded {max_bytes} bytes.",
                    )
                chunks.append(chunk)
            return URLFetchResponse(
                status_code=response.status_code,
                content=b"".join(chunks),
                content_type=response.headers.get("Content-Type"),
            )


def sanitize_worksite_filename(worksite_id: str) -> str:
    if not isinstance(worksite_id, str) or not worksite_id.strip():
        raise WorksiteManagementError(
            "INVALID_WORKSITE_ID",
            "Worksite ID is required before a filename can be generated.",
        )
    stem = _SAFE_FILENAME_RUN.sub("_", worksite_id.strip())
    stem = _REPEATED_UNDERSCORES.sub("_", stem).strip("._-")
    if not stem:
        stem = "worksite"
    return f"{stem}.json"


def _compact_payload(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        return {
            "worksite_id": None,
            "name": None,
            "authorized_workers": None,
            "required_ppe": None,
        }
    workers_value = raw.get("authorized_workers")
    workers: object
    if isinstance(workers_value, list):
        compact_workers: list[object] = []
        for worker in workers_value:
            if isinstance(worker, Mapping):
                compact_workers.append(
                    {
                        "worker_id": worker.get("worker_id"),
                        "name": worker.get("name"),
                        "embedding": worker.get("embedding"),
                    }
                )
            else:
                compact_workers.append(worker)
        workers = compact_workers
    else:
        workers = workers_value
    return {
        "worksite_id": raw.get("worksite_id"),
        "name": raw.get("name"),
        "authorized_workers": workers,
        "required_ppe": raw.get("required_ppe"),
    }


def _payload_from_worksite(worksite: Worksite) -> dict[str, object]:
    return {
        "worksite_id": worksite.worksite_id,
        "name": worksite.name,
        "authorized_workers": [
            {
                "worker_id": worker.worker_id,
                "name": worker.name,
                "embedding": list(worker.embedding),
            }
            for worker in worksite.authorized_workers
        ],
        "required_ppe": list(worksite.required_ppe),
    }


def _normalize_import_document(
    document: ImportDocument | Mapping[str, object],
) -> ImportDocument:
    if (
        isinstance(document, ImportDocument)
        and isinstance(document.source_name, str)
        and isinstance(document.content, (str, bytes))
    ):
        return document
    if isinstance(document, Mapping):
        source_name = document.get("source_name")
        content = document.get("content")
        if isinstance(source_name, str) and isinstance(content, (str, bytes)):
            return ImportDocument(source_name=source_name, content=content)
    raise WorksiteManagementError(
        "INVALID_IMPORT_DOCUMENT",
        "Each import document requires source_name and text or byte content.",
    )


def _document_text(content: str | bytes) -> str:
    return content if isinstance(content, str) else content.decode("utf-8")


def _read_text(path: Path) -> tuple[str, str | None]:
    try:
        content = path.read_bytes()
    except OSError as error:
        return "", f"Wo.No. file cannot be read: {error}"
    try:
        return content.decode("utf-8"), None
    except UnicodeDecodeError as error:
        return content.decode("utf-8", errors="replace"), str(error)


def _issue_summaries(issues: Sequence[ValidationIssue]) -> tuple[IssueSummary, ...]:
    return tuple(IssueSummary.from_validation_issue(issue) for issue in issues)


def _first_issue_message(issues: Sequence[IssueSummary], fallback: str) -> str:
    return issues[0].message if issues else fallback


def _atomic_write_text(path: Path, content: str, *, overwrite: bool) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        if not overwrite and (path.exists() or path.is_symlink()):
            raise WorksiteManagementError(
                "WORKSITE_FILENAME_COLLISION",
                f"Wo.No. destination {path.name!r} already exists and was not overwritten.",
            )
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except WorksiteManagementError:
        raise
    except OSError as error:
        raise WorksiteManagementError(
            "WORKSITE_WRITE_FAILED",
            f"Wo.No. file {path.name!r} could not be written atomically: {error}",
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
