from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import NoReturn

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from signatus_contracts import CameraStatus, ValidationReport, ValidationSeverity

from .ai_client import AIServiceClient, AIServiceClientError
from .controller import (
    CameraCommandError,
    CameraTransitionError,
    CoreController,
)
from .domain import GUIStatusSignal
from .enrollment import WorksiteEnrollmentError, materialize_worksite_embeddings
from .ppe import PPE_CLASS_MAP
from .settings import CoreSettings
from .worker_profiles import WorkerProfileRepository, validate_worker_profile
from .worksite_management import (
    ManagedWorksiteEntry,
    WorksiteManagementError,
    WorksiteManagementService,
)
from .worksites import WorksiteRepository


class GUISignalHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self.latest: GUIStatusSignal | None = None

    async def emit(self, signal: GUIStatusSignal) -> None:
        self.latest = signal
        dead: list[WebSocket] = []
        for client in tuple(self._clients):
            try:
                await client.send_json(signal.to_dict())
            except (OSError, RuntimeError, WebSocketDisconnect):
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    async def connect(self, socket: WebSocket) -> None:
        await socket.accept()
        self._clients.add(socket)
        if self.latest is not None:
            await socket.send_json(self.latest.to_dict())

    def disconnect(self, socket: WebSocket) -> None:
        self._clients.discard(socket)


settings = CoreSettings.from_environment()
worksites = WorksiteRepository(settings.worksite_dir)
worksite_manager = WorksiteManagementService(settings.worksite_dir, worksites)
worker_profiles = WorkerProfileRepository(settings.worker_profile_dir)
signals = GUISignalHub()
ai_client = AIServiceClient(settings.ai_base_url, settings.ai_events_url)
controller = CoreController(
    ai_commands=ai_client,
    signal_sink=signals,
    ppe_policy=PPE_CLASS_MAP,
    face_match_min_cosine_similarity=settings.face_match_min_cosine_similarity,
)
stop_event = asyncio.Event()


async def _synchronize_camera_status_forever() -> None:
    while not stop_event.is_set():
        try:
            await controller.synchronize_camera_status()
        except CameraCommandError:
            # Camera command/status failures are exposed as Camera ERROR. Core
            # and the AI event connection remain independently supervised.
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.5)
        except TimeoutError:
            pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    catalog = worksites.load_catalog()
    if catalog.has_fatal_errors:
        details = "; ".join(issue.message for issue in catalog.fatal_issues)
        raise RuntimeError(f"Fatal deployment validation failed: {details}")
    stop_event.clear()
    listener = asyncio.create_task(
        ai_client.listen_forever(controller.handle_event, stop_event),
        name="ai-event-listener",
    )
    camera_status_sync = asyncio.create_task(
        _synchronize_camera_status_forever(),
        name="ai-camera-status-sync",
    )
    try:
        yield
    finally:
        stop_event.set()
        listener.cancel()
        camera_status_sync.cancel()
        await asyncio.gather(listener, camera_status_sync, return_exceptions=True)
        await ai_client.close()


app = FastAPI(title="Signatus Core", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict[str, object]:
    camera = controller.camera_status
    catalog = worksites.load_catalog()
    return {
        "status": "error" if catalog.has_fatal_errors else "ok",
        "state": controller.state.value,
        "ai_events_connected": ai_client.events_connected,
        "camera_state": camera.state.value,
        "camera_error": camera.error,
        "validation_status": _validation_status(catalog.validation_report),
        "usable_worksite_count": len(catalog.available_worksites),
    }


@app.get("/api/camera", response_model=CameraStatus)
async def camera_status() -> CameraStatus:
    return controller.camera_status


@app.post("/api/camera/start", response_model=CameraStatus)
async def start_camera() -> CameraStatus:
    try:
        return await controller.start_camera()
    except CameraTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CameraCommandError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/camera/stop", response_model=CameraStatus)
async def stop_camera() -> CameraStatus:
    try:
        return await controller.stop_camera()
    except CameraTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CameraCommandError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/worksites")
async def list_worksites() -> list[dict[str, object]]:
    return [
        {
            "worksite_id": record.worksite_id,
            "name": record.name,
            "required_ppe": list(record.required_ppe),
            "available": record.available,
            "unavailable_reason": record.unavailable_reason,
            "valid_worker_count": record.valid_worker_count,
            "invalid_worker_count": record.invalid_worker_count,
        }
        for record in worksites.list_records()
        if record.worksite_id is not None and record.name is not None
    ]


@app.get("/api/validation")
async def validation_report() -> dict[str, object]:
    report = worksites.validation_report
    return {
        "status": _validation_status(report),
        "fatal_count": sum(
            issue.severity is ValidationSeverity.FATAL for issue in report.issues
        ),
        "data_error_count": sum(
            issue.severity is ValidationSeverity.DATA_ERROR for issue in report.issues
        ),
        "warning_count": sum(
            issue.severity is ValidationSeverity.WARNING for issue in report.issues
        ),
        **report.model_dump(mode="json"),
    }


@app.get("/api/worksite-manager")
async def manager_catalog() -> list[dict[str, object]]:
    return _serialize_manager_entries(worksite_manager.list_entries())


@app.post("/api/worksite-manager/refresh")
async def refresh_manager_catalog() -> list[dict[str, object]]:
    return _serialize_manager_entries(worksite_manager.refresh())


@app.get("/api/worksite-manager/options")
async def manager_options() -> dict[str, object]:
    return {"required_ppe": list(worksite_manager.ppe_options)}


@app.get("/api/worksite-manager/files/{source}/details")
async def manager_details(source: str) -> dict[str, object]:
    try:
        entry = worksite_manager.details(source)
    except WorksiteManagementError as error:
        _raise_management_error(error)
    return _serialize_manager_entry(entry)


@app.get("/api/worksite-manager/files/{source}/edit")
async def manager_edit_data(source: str) -> dict[str, object]:
    try:
        entry = worksite_manager.details(source)
    except WorksiteManagementError as error:
        _raise_management_error(error)
    if entry.worksite_id is None or entry.name is None or not entry.workers:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "WORKSITE_NOT_EDITABLE",
                "message": "This invalid Wo.No. cannot be edited with the structured editor.",
            },
        )
    value = _serialize_manager_entry(entry)
    value["invalid_worker_messages"] = [
        issue.message for issue in entry.invalid_worker_issues
    ]
    return value


@app.get("/api/worksite-manager/files/{source}/json")
async def manager_json(source: str) -> dict[str, object]:
    try:
        return worksite_manager.view_json(source).to_dict()
    except WorksiteManagementError as error:
        _raise_management_error(error)


@app.post("/api/worksite-manager/create")
async def manager_create(payload: dict[str, object]) -> dict[str, object]:
    existing_source = _existing_worksite_source(payload.get("worksite_id"))
    if existing_source is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "DUPLICATE_WORKSITE_ID",
                "message": f"Wo.No. {payload.get('worksite_id')!r} already exists.",
                "existing_source": existing_source,
            },
        )
    payload = await _materialize_enrollment_workers(payload)
    try:
        result = worksite_manager.create(payload)
    except WorksiteManagementError as error:
        _raise_management_error(error)
    return {**result.to_dict(), "active_policy_unchanged": False}


@app.put("/api/worksite-manager/files/{source}")
async def manager_edit(source: str, payload: dict[str, object]) -> dict[str, object]:
    payload = await _materialize_enrollment_workers(payload)
    try:
        result = worksite_manager.edit(source, payload)
    except WorksiteManagementError as error:
        _raise_management_error(error)
    active = _is_active_worksite(result.worksite_id, result.source)
    value = result.to_dict()
    value["active_policy_unchanged"] = active
    if active:
        value["message"] = (
            f"{result.message} The active in-memory policy is unchanged until this Wo.No. "
            "is selected again through the normal selector."
        )
    return value


@app.post("/api/worksite-manager/import-local")
async def manager_import_local(payload: dict[str, object]) -> dict[str, object]:
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_IMPORT_BATCH",
                "message": "Local import requires a documents list.",
            },
        )
    return worksite_manager.import_documents(documents).to_dict()


@app.post("/api/worksite-manager/import-url")
async def manager_import_url(payload: dict[str, object]) -> dict[str, object]:
    url = payload.get("url")
    if not isinstance(url, str):
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_IMPORT_URL", "message": "A direct JSON URL is required."},
        )
    try:
        result = await worksite_manager.import_url(url)
    except WorksiteManagementError as error:
        _raise_management_error(error)
    if result.status == "IMPORTED":
        return {
            "imported": [result.to_dict()],
            "skipped": [],
            "failed": [],
            "imported_count": 1,
            "skipped_count": 0,
            "failed_count": 0,
        }
    return {
        "imported": [],
        "skipped": [result.to_dict()] if result.status == "SKIPPED" else [],
        "failed": [result.to_dict()] if result.status == "FAILED" else [],
        "imported_count": 0,
        "skipped_count": int(result.status == "SKIPPED"),
        "failed_count": int(result.status == "FAILED"),
    }


@app.delete("/api/worksite-manager/files/{source}")
async def manager_delete(source: str, confirmed: bool = False) -> dict[str, object]:
    active = controller.selected_worksite
    try:
        result = worksite_manager.delete(
            source,
            confirmed=confirmed,
            active_worksite_id=None if active is None else active.worksite_id,
            active_source=controller.selected_worksite_source,
        )
    except WorksiteManagementError as error:
        _raise_management_error(error)
    return result.to_dict()


@app.post("/api/worker-profiles/validate")
async def validate_profile(payload: dict[str, object]) -> dict[str, object]:
    result = validate_worker_profile(payload)
    if not result.valid or result.profile is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_WORKER_PROFILE",
                "message": "Worker Profile validation failed.",
                "errors": [issue.to_dict() for issue in result.errors],
            },
        )
    return result.profile.to_dict()


@app.post("/api/worker-profiles/create")
async def create_profile(payload: dict[str, object]) -> dict[str, object]:
    result = worker_profiles.create(payload)
    if not result.success:
        _raise_profile_save_error(result.errors)
    return result.to_dict()


@app.put("/api/worker-profiles/{source}")
async def edit_profile(source: str, payload: dict[str, object]) -> dict[str, object]:
    result = worker_profiles.edit(source, payload)
    if not result.success:
        _raise_profile_save_error(result.errors)
    return result.to_dict()


@app.get("/api/worker-profiles/{source}")
async def get_profile(source: str) -> dict[str, object]:
    record = worker_profiles.load(source)
    if not record.valid or record.profile is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "WORKER_PROFILE_UNAVAILABLE",
                "message": "Worker Profile is unavailable.",
                "errors": [issue.to_dict() for issue in record.errors],
            },
        )
    return {**record.profile.to_dict(), "source": record.source}


@app.post("/api/worksites/{worksite_id}/select")
async def select_worksite(worksite_id: str) -> dict[str, object]:
    record = worksites.get_record(worksite_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Worksite not found")
    if not record.available or record.worksite is None:
        raise HTTPException(
            status_code=409,
            detail=record.unavailable_reason or "Worksite configuration is unavailable",
        )
    worksite = record.worksite
    controller.select_worksite(worksite, source=record.source)
    return {
        "worksite_id": worksite.worksite_id,
        "name": worksite.name,
        "required_ppe": list(worksite.required_ppe),
        "available": True,
        "valid_worker_count": record.valid_worker_count,
        "invalid_worker_count": record.invalid_worker_count,
        "state": controller.state.value,
    }


@app.get("/api/state")
async def state() -> dict[str, object]:
    worksite = controller.selected_worksite
    camera = controller.camera_status
    return {
        "state": controller.state.value,
        "camera": camera.model_dump(mode="json"),
        "camera_state": camera.state.value,
        "camera_error": camera.error,
        "selected_worksite": None
        if worksite is None
        else {"worksite_id": worksite.worksite_id, "name": worksite.name},
        "latest_signal": None if signals.latest is None else signals.latest.to_dict(),
    }


@app.websocket("/ws/gui")
async def gui_events(socket: WebSocket) -> None:
    await signals.connect(socket)
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        signals.disconnect(socket)


def _validation_status(report: ValidationReport) -> str:
    issues = report.issues
    if any(issue.severity is ValidationSeverity.FATAL for issue in issues):
        return "FAIL"
    if any(issue.severity is ValidationSeverity.DATA_ERROR for issue in issues):
        return "PASS_WITH_DATA_ERRORS"
    return "PASS"


async def _materialize_enrollment_workers(
    payload: dict[str, object],
) -> dict[str, object]:
    try:
        materialized = await materialize_worksite_embeddings(payload, ai_client)
    except WorksiteEnrollmentError as error:
        raise HTTPException(status_code=422, detail=error.to_dict()) from error
    except AIServiceClientError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "AI_PROFILE_EMBEDDING_UNAVAILABLE",
                "message": "AI could not generate the Worker Profile embedding.",
            },
        ) from error
    if not isinstance(materialized, dict):
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_WORKSITE", "message": "Wo.No. must be a JSON object."},
        )
    return materialized


def _existing_worksite_source(worksite_id: object) -> str | None:
    if not isinstance(worksite_id, str) or not worksite_id.strip():
        return None
    try:
        entries = worksite_manager.list_entries(refresh=True)
    except WorksiteManagementError:
        return None
    return next(
        (entry.source for entry in entries if entry.worksite_id == worksite_id),
        None,
    )


def _serialize_manager_entries(
    entries: tuple[ManagedWorksiteEntry, ...],
) -> list[dict[str, object]]:
    return [_serialize_manager_entry(entry) for entry in entries]


def _serialize_manager_entry(entry: ManagedWorksiteEntry) -> dict[str, object]:
    active = controller.selected_worksite
    return entry.to_dict(
        active_worksite_id=None if active is None else active.worksite_id,
        active_source=controller.selected_worksite_source,
    )


def _is_active_worksite(worksite_id: str | None, source: str | None) -> bool:
    active = controller.selected_worksite
    return (
        source is not None and source == controller.selected_worksite_source
    ) or (
        active is not None
        and worksite_id is not None
        and worksite_id == active.worksite_id
    )


def _raise_management_error(error: WorksiteManagementError) -> NoReturn:
    not_found = {"WORKSITE_NOT_FOUND", "WORKSITE_NOT_IN_CATALOG"}
    conflicts = {
        "ACTIVE_WORKSITE_DELETE_FORBIDDEN",
        "DELETE_CONFIRMATION_REQUIRED",
        "DUPLICATE_WORKSITE_ID",
        "IMMUTABLE_WORKSITE_ID",
        "WORKSITE_FILENAME_COLLISION",
    }
    unavailable = {
        "WORKSITE_DIRECTORY_UNAVAILABLE",
        "WORKSITE_DIRECTORY_UNREADABLE",
        "WORKSITE_WRITE_FAILED",
        "WORKSITE_DELETE_FAILED",
    }
    if error.code in not_found:
        status = 404
    elif error.code in conflicts:
        status = 409
    elif error.code in unavailable:
        status = 503
    else:
        status = 422
    detail = error.to_dict()
    if error.code == "DUPLICATE_WORKSITE_ID":
        existing = next(
            (
                entry.source
                for entry in worksite_manager.list_entries(refresh=True)
                if entry.worksite_id is not None and entry.worksite_id in error.message
            ),
            None,
        )
        if existing is not None:
            detail["existing_source"] = existing
    raise HTTPException(status_code=status, detail=detail)


def _raise_profile_save_error(errors: tuple[object, ...]) -> NoReturn:
    serialized = [error.to_dict() for error in errors if hasattr(error, "to_dict")]
    codes = {item.get("code") for item in serialized}
    status = 409 if {"DUPLICATE_WORKER_ID", "IMMUTABLE_WORKER_ID"} & codes else 422
    message = serialized[0]["message"] if serialized else "Worker Profile could not be saved."
    raise HTTPException(
        status_code=status,
        detail={
            "code": next(iter(codes), "WORKER_PROFILE_SAVE_FAILED"),
            "message": message,
            "errors": serialized,
        },
    )
