from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from signatus_core.ppe import PPE_CLASS_MAP
from signatus_core.worksite_management import (
    ImportDocument,
    URLFetchResponse,
    WorksiteManagementError,
    WorksiteManagementService,
    sanitize_worksite_filename,
)
from signatus_core.worksites import WorksiteRepository


def _embedding(axis: int = 0) -> list[float]:
    values = [0.0] * 128
    values[axis] = 1.0
    return values


def _worker(worker_id: str = "W001", *, axis: int = 0) -> dict[str, object]:
    return {
        "worker_id": worker_id,
        "name": f"Worker {worker_id}",
        "embedding": _embedding(axis),
    }


def _worksite(
    worksite_id: str = "WO-001",
    *,
    name: str | None = None,
    workers: list[object] | None = None,
    required_ppe: list[object] | None = None,
) -> dict[str, object]:
    return {
        "worksite_id": worksite_id,
        "name": f"Worksite {worksite_id}" if name is None else name,
        "authorized_workers": [_worker()] if workers is None else workers,
        "required_ppe": ["helmet"] if required_ppe is None else required_ppe,
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class _CountingRepository(WorksiteRepository):
    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self.refresh_calls = 0

    def refresh_catalog(self):  # type: ignore[no-untyped-def]
        self.refresh_calls += 1
        return super().refresh_catalog()


class WorksiteManagementTests(unittest.TestCase):
    def test_ppe_options_come_from_core_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = WorksiteManagementService(Path(directory))

            self.assertEqual(service.ppe_options, tuple(PPE_CLASS_MAP))

    def test_sanitized_filename_keeps_internal_id_out_of_path_handling(self) -> None:
        self.assertEqual(
            sanitize_worksite_filename("PMII / WO 015"),
            "PMII_WO_015.json",
        )
        self.assertEqual(sanitize_worksite_filename("../../WO:*? 15"), "WO_15.json")

    def test_generator_create_writes_compact_valid_json_and_allows_zero_ppe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(root)
            payload = _worksite(
                "PMII / WO 015",
                name="Pump maintenance",
                required_ppe=[],
            )
            payload["ignored_enterprise_field"] = "not persisted"

            result = service.create(payload)

            self.assertEqual(result.status, "CREATED")
            self.assertEqual(result.source, "PMII_WO_015.json")
            stored = json.loads((root / result.source).read_text(encoding="utf-8"))
            self.assertEqual(
                tuple(stored),
                ("worksite_id", "name", "authorized_workers", "required_ppe"),
            )
            self.assertEqual(stored["worksite_id"], "PMII / WO 015")
            self.assertEqual(stored["required_ppe"], [])
            self.assertEqual(set(stored["authorized_workers"][0]), {"worker_id", "name", "embedding"})
            self.assertEqual(len(stored["authorized_workers"][0]["embedding"]), 128)
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_create_rejects_required_fields_zero_workers_and_bad_worker(self) -> None:
        invalid_cases = (
            (_worksite(""), "INVALID_WORKSITE_ID"),
            (_worksite(name=""), "INVALID_WORKSITE_NAME"),
            (_worksite(workers=[]), "NO_USABLE_AUTHORIZED_WORKERS"),
            (
                _worksite(
                    workers=[
                        {
                            "worker_id": "BAD",
                            "name": "Bad Worker",
                            "embedding": [1.0, 0.0, 0.0],
                        }
                    ]
                ),
                "INVALID_SFACE_DESCRIPTOR_LENGTH",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(root)

            for payload, expected_issue in invalid_cases:
                with self.subTest(expected_issue=expected_issue):
                    with self.assertRaises(WorksiteManagementError) as raised:
                        service.create(payload)
                    self.assertEqual(raised.exception.code, "INVALID_WORKSITE")
                    self.assertIn(
                        expected_issue,
                        {issue.code for issue in raised.exception.issues},
                    )

            self.assertEqual(list(root.glob("*.json")), [])

    def test_create_rejects_duplicate_workers_even_when_another_worker_is_valid(self) -> None:
        payload = _worksite(
            workers=[
                _worker("DUP", axis=0),
                _worker("DUP", axis=1),
                _worker("GOOD", axis=2),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            service = WorksiteManagementService(Path(directory))

            with self.assertRaises(WorksiteManagementError) as raised:
                service.create(payload)

        self.assertIn("DUPLICATE_WORKER_ID", {issue.code for issue in raised.exception.issues})

    def test_duplicate_id_scan_uses_internal_id_not_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "unrelated.synthetic-valid.json"
            _write(original, _worksite("WO-015"))
            original_text = original.read_text(encoding="utf-8")
            service = WorksiteManagementService(root)

            with self.assertRaises(WorksiteManagementError) as raised:
                service.create(_worksite("WO-015", name="Replacement"))

            self.assertEqual(raised.exception.code, "DUPLICATE_WORKSITE_ID")
            self.assertIn(original.name, raised.exception.message)
            self.assertEqual(original.read_text(encoding="utf-8"), original_text)

    def test_sanitized_filename_collision_does_not_overwrite_other_worksite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collision = root / "PMII_WO_015.json"
            _write(collision, _worksite("ANOTHER-ID"))
            before = collision.read_bytes()
            service = WorksiteManagementService(root)

            with self.assertRaises(WorksiteManagementError) as raised:
                service.create(_worksite("PMII / WO 015"))

            self.assertEqual(raised.exception.code, "WORKSITE_FILENAME_COLLISION")
            self.assertEqual(collision.read_bytes(), before)

    def test_editor_keeps_worksite_id_immutable_and_saves_valid_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "odd-source-name.json"
            _write(source, _worksite("WO-EDIT"))
            service = WorksiteManagementService(root)

            changed_id = _worksite("WO-CHANGED", name="Changed")
            with self.assertRaises(WorksiteManagementError) as raised:
                service.edit(source.name, changed_id)
            self.assertEqual(raised.exception.code, "IMMUTABLE_WORKSITE_ID")

            edited = _worksite(
                "WO-EDIT",
                name="New site name",
                workers=[_worker("W900", axis=4)],
                required_ppe=["boots", "goggles"],
            )
            result = service.edit(source.name, edited)

            self.assertEqual(result.status, "UPDATED")
            stored = json.loads(source.read_text(encoding="utf-8"))
            self.assertEqual(stored["worksite_id"], "WO-EDIT")
            self.assertEqual(stored["name"], "New site name")
            self.assertEqual(stored["authorized_workers"][0]["worker_id"], "W900")
            self.assertEqual(stored["required_ppe"], ["boots", "goggles"])

    def test_list_includes_partial_and_malformed_entries_with_details_and_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root / "partial.json",
                _worksite(
                    "WO-PARTIAL",
                    workers=[
                        _worker("GOOD"),
                        {
                            "worker_id": "BAD",
                            "name": "Bad Worker",
                            "embedding": [1.0, 0.0, 0.0],
                        },
                    ],
                ),
            )
            malformed = root / "broken.json"
            malformed.write_text('{"worksite_id": "BROKEN",', encoding="utf-8")
            service = WorksiteManagementService(root)

            entries = {entry.source: entry for entry in service.refresh()}

            self.assertEqual(set(entries), {"partial.json", "broken.json"})
            partial = entries["partial.json"]
            self.assertTrue(partial.available)
            self.assertEqual(partial.validity_state, "PARTIAL")
            self.assertEqual(partial.valid_worker_count, 1)
            self.assertEqual(partial.invalid_worker_count, 1)
            self.assertEqual(partial.workers[0].name, "Worker GOOD")
            self.assertEqual(len(partial.workers[0].embedding), 128)
            self.assertEqual(
                {issue.code for issue in partial.invalid_worker_issues},
                {"INVALID_SFACE_DESCRIPTOR_LENGTH"},
            )
            broken = entries["broken.json"]
            self.assertFalse(broken.available)
            self.assertIn('"worksite_id"', broken.raw_json)
            self.assertIsNotNone(broken.parse_error)
            self.assertIn("INVALID_WORKSITE_JSON", {issue.code for issue in broken.issues})

            serialized = partial.to_dict(active_worksite_id="WO-PARTIAL")
            self.assertTrue(serialized["active"])
            self.assertEqual(serialized["authorized_workers"][0]["name"], "Worker GOOD")

    def test_view_json_is_read_only_formats_valid_and_preserves_malformed_raw_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            malformed = root / "malformed.json"
            _write(valid, _worksite())
            malformed.write_text("{not-json", encoding="utf-8")
            valid_before = valid.read_bytes()
            malformed_before = malformed.read_bytes()
            service = WorksiteManagementService(root)

            valid_view = service.view_json(valid.name)
            malformed_view = service.view_json(malformed.name)

            self.assertTrue(valid_view.read_only)
            self.assertIsNotNone(valid_view.formatted)
            self.assertIsNone(valid_view.parse_error)
            self.assertEqual(malformed_view.raw, "{not-json")
            self.assertIsNone(malformed_view.formatted)
            self.assertIsNotNone(malformed_view.parse_error)
            self.assertEqual(valid.read_bytes(), valid_before)
            self.assertEqual(malformed.read_bytes(), malformed_before)

    def test_source_resolution_rejects_traversal_backslashes_and_non_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = WorksiteManagementService(Path(directory))

            for unsafe in ("../secret.json", "sub/file.json", "sub\\file.json", "notes.txt"):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(WorksiteManagementError) as raised:
                        service.view_json(unsafe)
                    self.assertEqual(raised.exception.code, "UNSAFE_WORKSITE_SOURCE")

    def test_local_batch_processes_each_document_and_reports_all_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "already-here.weird-name.json", _worksite("WO-DUP"))
            service = WorksiteManagementService(root)
            documents = (
                ImportDocument("first.json", json.dumps(_worksite("WO-ONE"))),
                ImportDocument("broken.json", "{broken"),
                ImportDocument("duplicate.json", json.dumps(_worksite("WO-DUP"))),
                ImportDocument("last.json", json.dumps(_worksite("WO-TWO"))),
            )

            result = service.import_documents(documents)

            self.assertEqual(
                [item.worksite_id for item in result.imported],
                ["WO-ONE", "WO-TWO"],
            )
            self.assertEqual([item.worksite_id for item in result.skipped], ["WO-DUP"])
            self.assertEqual([item.source for item in result.failed], ["broken.json"])
            self.assertTrue((root / "WO-ONE.json").is_file())
            self.assertTrue((root / "WO-TWO.json").is_file())
            self.assertEqual(result.to_dict()["imported_count"], 2)

    def test_bad_document_envelope_does_not_cancel_rest_of_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(root)

            result = service.import_documents(
                (
                    {"source_name": "bad-envelope.json", "content": 123},
                    {
                        "source_name": "good.json",
                        "content": json.dumps(_worksite("WO-GOOD")),
                    },
                )
            )

            self.assertEqual(len(result.failed), 1)
            self.assertEqual(result.failed[0].source, "bad-envelope.json")
            self.assertEqual(len(result.imported), 1)
            self.assertTrue((root / "WO-GOOD.json").exists())

    def test_partial_worker_import_filters_bad_worker_and_reports_it(self) -> None:
        payload = _worksite(
            "WO-PARTIAL",
            workers=[
                _worker("GOOD"),
                {
                    "worker_id": "BAD",
                    "name": "Bad Worker",
                    "embedding": [1.0, 0.0, 0.0],
                },
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(root)

            result = service.import_documents(
                (ImportDocument("partial.json", json.dumps(payload)),)
            )

            self.assertEqual(len(result.imported), 1)
            imported = result.imported[0]
            self.assertEqual(
                {issue.code for issue in imported.skipped_workers},
                {"INVALID_SFACE_DESCRIPTOR_LENGTH"},
            )
            stored = json.loads((root / "WO-PARTIAL.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [worker["worker_id"] for worker in stored["authorized_workers"]],
                ["GOOD"],
            )

    def test_import_rejects_document_when_zero_valid_workers_remain(self) -> None:
        payload = _worksite(
            "WO-NONE",
            workers=[
                {
                    "worker_id": "BAD",
                    "name": "Bad Worker",
                    "embedding": [1.0, 0.0, 0.0],
                }
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(root)

            result = service.import_documents(
                (ImportDocument("none.json", json.dumps(payload)),)
            )

            self.assertEqual(len(result.failed), 1)
            self.assertIn(
                "NO_USABLE_AUTHORIZED_WORKERS",
                {issue.code for issue in result.failed[0].issues},
            )
            self.assertFalse((root / "WO-NONE.json").exists())

    def test_valid_worker_profile_produces_transient_face_image_worker(self) -> None:
        image = b"\x89PNG\r\n\x1a\n" + b"profile-image"
        face_image = "data:image/png;base64," + base64.b64encode(image).decode()
        profile = {
            "worker_id": "PROFILE-W1",
            "name": "Profile Worker",
            "face_image": face_image,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(root)

            validated = service.validate_worker_profile(profile)
            worker = service.worker_from_profile(profile)

            self.assertEqual(validated["face_image"], face_image)
            self.assertEqual(worker["face_image"], face_image)
            self.assertNotIn("embedding", worker)

    def test_invalid_worker_profiles_are_rejected(self) -> None:
        image = b"\x89PNG\r\n\x1a\n" + b"profile-image"
        face_image = "data:image/png;base64," + base64.b64encode(image).decode()
        cases = (
            {
                "worker_id": "W1",
                "name": "Worker",
                "face_image": "not-a-data-uri",
            },
            {
                "worker_id": "W1",
                "name": "Worker",
                "face_image": face_image,
                "embedding": _embedding(),
            },
            {
                "worker_id": "",
                "name": "Worker",
                "face_image": face_image,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            service = WorksiteManagementService(Path(directory))

            for profile in cases:
                with self.subTest(profile=profile), self.assertRaises(
                    WorksiteManagementError
                ):
                    service.validate_worker_profile(profile)

    def test_delete_requires_confirmation_and_refuses_active_id_or_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stored.json"
            _write(source, _worksite("WO-ACTIVE"))
            service = WorksiteManagementService(root)

            with self.assertRaises(WorksiteManagementError) as confirmation:
                service.delete(source.name, confirmed=False)
            self.assertEqual(confirmation.exception.code, "DELETE_CONFIRMATION_REQUIRED")

            with self.assertRaises(WorksiteManagementError) as active_id:
                service.delete(
                    source.name,
                    confirmed=True,
                    active_worksite_id="WO-ACTIVE",
                )
            self.assertEqual(active_id.exception.code, "ACTIVE_WORKSITE_DELETE_FORBIDDEN")

            with self.assertRaises(WorksiteManagementError) as active_source:
                service.delete(
                    source.name,
                    confirmed=True,
                    active_source=source.name,
                )
            self.assertEqual(active_source.exception.code, "ACTIVE_WORKSITE_DELETE_FORBIDDEN")

            result = service.delete(
                source.name,
                confirmed=True,
                active_worksite_id="WO-OTHER",
            )

            self.assertEqual(result.status, "DELETED")
            self.assertFalse(source.exists())

    def test_explicit_refresh_sees_external_file_without_changing_any_active_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(root)
            self.assertEqual(service.list_entries(), ())
            _write(root / "external.json", _worksite("WO-EXTERNAL"))

            self.assertEqual(service.list_entries(), ())
            refreshed = service.refresh()

            self.assertEqual([entry.worksite_id for entry in refreshed], ["WO-EXTERNAL"])

    def test_repository_refresh_occurs_only_after_successful_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _CountingRepository(root)
            service = WorksiteManagementService(root, repository)

            with self.assertRaises(WorksiteManagementError):
                service.create(_worksite(workers=[]))
            service.import_documents((ImportDocument("broken.json", "{broken"),))
            self.assertEqual(repository.refresh_calls, 0)

            service.create(_worksite("WO-SUCCESS"))
            self.assertEqual(repository.refresh_calls, 1)


class WorksiteURLImportTests(unittest.IsolatedAsyncioTestCase):
    async def test_https_import_uses_timeout_and_size_limit_and_stores_json(self) -> None:
        calls: list[tuple[str, float, int]] = []

        async def fetcher(url: str, timeout: float, max_bytes: int) -> URLFetchResponse:
            calls.append((url, timeout, max_bytes))
            return URLFetchResponse(
                status_code=200,
                content=json.dumps(_worksite("WO-URL")),
                content_type="application/json",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(
                root,
                fetcher=fetcher,
                url_timeout_seconds=3.5,
                url_max_bytes=50_000,
            )

            result = await service.import_url("https://pmii.example/direct-worksite.json")

            self.assertEqual(result.status, "IMPORTED")
            self.assertEqual(calls, [("https://pmii.example/direct-worksite.json", 3.5, 50_000)])
            self.assertTrue((root / "WO-URL.json").exists())

    async def test_non_http_urls_and_url_credentials_are_rejected(self) -> None:
        async def unused_fetcher(
            _url: str, _timeout: float, _max_bytes: int
        ) -> URLFetchResponse:
            raise AssertionError("fetcher must not be called")

        with tempfile.TemporaryDirectory() as directory:
            service = WorksiteManagementService(Path(directory), fetcher=unused_fetcher)

            for url, code in (
                ("file:///tmp/worksite.json", "INVALID_IMPORT_URL"),
                ("ftp://example.test/worksite.json", "INVALID_IMPORT_URL"),
                ("https://user:password@example.test/worksite.json", "URL_AUTHENTICATION_UNSUPPORTED"),
            ):
                with self.subTest(url=url):
                    with self.assertRaises(WorksiteManagementError) as raised:
                        await service.import_url(url)
                    self.assertEqual(raised.exception.code, code)

    async def test_http_error_oversize_and_invalid_json_are_nonfatal(self) -> None:
        responses: list[URLFetchResponse] = [
            URLFetchResponse(503, b"unavailable"),
            URLFetchResponse(200, b"x" * 21),
            URLFetchResponse(200, b"{broken"),
            URLFetchResponse(200, json.dumps(_worksite("WO-AFTER-ERROR"))),
        ]

        async def fetcher(
            _url: str, _timeout: float, _max_bytes: int
        ) -> URLFetchResponse:
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = WorksiteManagementService(
                root,
                fetcher=fetcher,
                url_max_bytes=20,
            )

            with self.assertRaises(WorksiteManagementError) as http_error:
                await service.import_url("https://example.test/status")
            self.assertEqual(http_error.exception.code, "URL_HTTP_ERROR")

            with self.assertRaises(WorksiteManagementError) as oversize:
                await service.import_url("https://example.test/large")
            self.assertEqual(oversize.exception.code, "URL_RESPONSE_TOO_LARGE")

            invalid = await service.import_url("https://example.test/broken")
            self.assertEqual(invalid.status, "FAILED")
            self.assertFalse(list(root.glob("*.json")))

            service_with_room = WorksiteManagementService(
                root,
                fetcher=fetcher,
                url_max_bytes=20_000,
            )
            recovered = await service_with_room.import_url("https://example.test/good")
            self.assertEqual(recovered.status, "IMPORTED")
            self.assertTrue((root / "WO-AFTER-ERROR.json").exists())

    async def test_fetch_timeout_is_scoped_management_error(self) -> None:
        async def fetcher(_url: str, _timeout: float, _max_bytes: int) -> bytes:
            raise TimeoutError

        with tempfile.TemporaryDirectory() as directory:
            service = WorksiteManagementService(Path(directory), fetcher=fetcher)

            with self.assertRaises(WorksiteManagementError) as raised:
                await service.import_url("https://example.test/slow.json")

            self.assertEqual(raised.exception.code, "URL_IMPORT_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
