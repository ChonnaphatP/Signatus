from __future__ import annotations

import json
from collections.abc import Callable
from urllib.parse import quote, urlsplit, urlunsplit

from PySide6.QtCore import QByteArray, QObject, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWebSockets import QWebSocket

from .contracts import (
    Outcome,
    Worksite,
    parse_camera_status,
    parse_import_summary,
    parse_manager_catalog,
    parse_outcome,
    parse_raw_json_document,
    parse_validation_report,
    parse_worker_profile,
    parse_worksite_draft,
    parse_worksites,
)


class CoreClient(QObject):
    worksites_loaded = Signal(object)
    worksite_selected = Signal(object)
    outcome_received = Signal(object)
    request_failed = Signal(str)
    protocol_error = Signal(str)
    connection_changed = Signal(bool)
    state_changed = Signal(str)
    camera_status_changed = Signal(object)
    camera_stop_completed = Signal(object)
    camera_stop_failed = Signal(str)
    validation_report_loaded = Signal(object)
    manager_catalog_loaded = Signal(object)
    manager_options_loaded = Signal(object)
    manager_draft_loaded = Signal(object)
    manager_json_loaded = Signal(object)
    manager_details_loaded = Signal(object)
    manager_changed = Signal(object)
    manager_import_completed = Signal(object)
    management_failed = Signal(object)
    worker_profile_validated = Signal(object)
    worker_profile_saved = Signal(object)

    def __init__(self, core_url: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._base_url = core_url.rstrip("/")
        self._network = QNetworkAccessManager(self)
        self._socket = QWebSocket(parent=self)
        self._reconnect = QTimer(self)
        self._reconnect.setSingleShot(True)
        self._reconnect.setInterval(1_000)
        self._reconnect.timeout.connect(self.connect_outcomes)
        self._state_poll = QTimer(self)
        self._state_poll.setInterval(500)
        self._state_poll.timeout.connect(self._poll_state)
        self._state_reply_pending = False
        self._socket.connected.connect(lambda: self.connection_changed.emit(True))
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.textMessageReceived.connect(self._on_message)
        self._socket.errorOccurred.connect(self._on_socket_error)

    def load_worksites(self) -> None:
        self._get_json("/api/worksites", self._handle_worksites)

    def load_validation_report(self) -> None:
        self._get_json("/api/validation", self._handle_validation_report)

    def select_worksite(self, worksite: Worksite) -> None:
        path = f"/api/worksites/{quote(worksite.worksite_id, safe='')}/select"
        request = QNetworkRequest(QUrl(f"{self._base_url}{path}"))
        reply = self._network.post(request, QByteArray())
        reply.finished.connect(lambda: self._finish_json(reply, self._handle_selected))

    def start_camera(self) -> None:
        self._post_json("/api/camera/start", self._handle_camera_status)

    def stop_camera(self) -> None:
        self._post_json(
            "/api/camera/stop",
            self._handle_stop_camera_status,
            self.camera_stop_failed.emit,
        )

    def load_manager(self, *, refresh: bool = False) -> None:
        method = "POST" if refresh else "GET"
        self._management_request(
            method,
            "/api/worksite-manager/refresh" if refresh else "/api/worksite-manager",
            callback=self._handle_manager_catalog,
        )

    def load_manager_options(self) -> None:
        self._management_request(
            "GET",
            "/api/worksite-manager/options",
            callback=self._handle_manager_options,
        )

    def load_manager_draft(self, source: str) -> None:
        self._management_request(
            "GET",
            f"/api/worksite-manager/files/{quote(source, safe='')}/edit",
            callback=self._handle_manager_draft,
        )

    def load_manager_details(self, source: str) -> None:
        self._management_request(
            "GET",
            f"/api/worksite-manager/files/{quote(source, safe='')}/details",
            callback=lambda payload: self.manager_details_loaded.emit(payload),
        )

    def load_manager_json(self, source: str) -> None:
        self._management_request(
            "GET",
            f"/api/worksite-manager/files/{quote(source, safe='')}/json",
            callback=self._handle_manager_json,
        )

    def create_worksite(self, payload: dict[str, object]) -> None:
        self._management_request(
            "POST",
            "/api/worksite-manager/create",
            payload,
            self.manager_changed.emit,
        )

    def edit_worksite(self, source: str, payload: dict[str, object]) -> None:
        self._management_request(
            "PUT",
            f"/api/worksite-manager/files/{quote(source, safe='')}",
            payload,
            self.manager_changed.emit,
        )

    def delete_worksite(self, source: str) -> None:
        self._management_request(
            "DELETE",
            f"/api/worksite-manager/files/{quote(source, safe='')}?confirmed=true",
            callback=self.manager_changed.emit,
        )

    def import_worksite_documents(self, documents: list[dict[str, str]]) -> None:
        self._management_request(
            "POST",
            "/api/worksite-manager/import-local",
            {"documents": documents},
            self._handle_manager_import,
        )

    def import_worksite_url(self, url: str) -> None:
        self._management_request(
            "POST",
            "/api/worksite-manager/import-url",
            {"url": url},
            self._handle_manager_import,
        )

    def validate_worker_profile(self, payload: object) -> None:
        self._management_request(
            "POST",
            "/api/worker-profiles/validate",
            payload,
            self._handle_worker_profile_validated,
        )

    def create_worker_profile(self, payload: dict[str, object]) -> None:
        self._management_request(
            "POST",
            "/api/worker-profiles/create",
            payload,
            self._handle_worker_profile_saved,
        )

    def edit_worker_profile(self, source: str, payload: dict[str, object]) -> None:
        self._management_request(
            "PUT",
            f"/api/worker-profiles/{quote(source, safe='')}",
            payload,
            self._handle_worker_profile_saved,
        )

    def connect_outcomes(self) -> None:
        if self._socket.isValid():
            return
        self._socket.open(QUrl(self._websocket_url()))

    def close(self) -> None:
        self._reconnect.stop()
        self._state_poll.stop()
        self._socket.close()

    def start_state_monitoring(self) -> None:
        if not self._state_poll.isActive():
            self._state_poll.start()
        self._poll_state()

    def _get_json(self, path: str, callback: Callable[[object], None]) -> None:
        reply = self._network.get(QNetworkRequest(QUrl(f"{self._base_url}{path}")))
        reply.finished.connect(lambda: self._finish_json(reply, callback))

    def _post_json(
        self,
        path: str,
        callback: Callable[[object], None],
        failure_callback: Callable[[str], None] | None = None,
    ) -> None:
        request = QNetworkRequest(QUrl(f"{self._base_url}{path}"))
        reply = self._network.post(request, QByteArray())
        reply.finished.connect(
            lambda: self._finish_json(reply, callback, failure_callback)
        )

    def _management_request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        callback: Callable[[object], None] | None = None,
    ) -> None:
        request = QNetworkRequest(QUrl(f"{self._base_url}{path}"))
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        try:
            body = QByteArray(
                b"" if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
            )
        except (TypeError, ValueError) as error:
            self.management_failed.emit({"message": str(error)})
            return
        reply = self._network.sendCustomRequest(request, method.encode("ascii"), body)
        reply.finished.connect(
            lambda: self._finish_management_json(reply, callback or (lambda _payload: None))
        )

    def _finish_management_json(
        self,
        reply: QNetworkReply,
        callback: Callable[[object], None],
    ) -> None:
        try:
            raw_body = bytes(reply.readAll())
            status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if reply.error() != QNetworkReply.NetworkError.NoError or not (
                isinstance(status, int) and 200 <= status < 300
            ):
                detail: object = reply.errorString()
                try:
                    error_payload = json.loads(raw_body)
                    if isinstance(error_payload, dict):
                        detail = error_payload.get("detail", detail)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
                message = detail.get("message") if isinstance(detail, dict) else detail
                self.management_failed.emit(
                    {"status": status, "message": str(message), "detail": detail}
                )
                return
            try:
                payload = json.loads(raw_body or b"{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.management_failed.emit({"status": status, "message": "Core returned invalid JSON"})
                return
            callback(payload)
        finally:
            reply.deleteLater()

    def _finish_json(
        self,
        reply: QNetworkReply,
        callback: Callable[[object], None],
        failure_callback: Callable[[str], None] | None = None,
    ) -> None:
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                message = reply.errorString()
                self.request_failed.emit(message)
                if failure_callback is not None:
                    failure_callback(message)
                return
            status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if not isinstance(status, int) or not 200 <= status < 300:
                message = f"Core returned HTTP {status}"
                self.request_failed.emit(message)
                if failure_callback is not None:
                    failure_callback(message)
                return
            try:
                payload = json.loads(bytes(reply.readAll()))
            except (UnicodeDecodeError, json.JSONDecodeError):
                message = "Core returned invalid JSON"
                self.protocol_error.emit(message)
                if failure_callback is not None:
                    failure_callback(message)
                return
            callback(payload)
        finally:
            reply.deleteLater()

    def _handle_worksites(self, payload: object) -> None:
        try:
            worksites = parse_worksites(payload)
        except ValueError as error:
            self.protocol_error.emit(str(error))
            return
        self.worksites_loaded.emit(worksites)

    def _handle_selected(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self.protocol_error.emit("Core returned an invalid worksite selection")
            return
        try:
            selected = parse_worksites([payload])[0]
        except (KeyError, ValueError):
            self.protocol_error.emit("Core returned an incomplete worksite selection")
            return
        self.worksite_selected.emit(selected)

    def _handle_camera_status(self, payload: object) -> None:
        try:
            status = parse_camera_status(payload)
        except ValueError as error:
            self.protocol_error.emit(str(error))
            return
        self.camera_status_changed.emit(status)

    def _handle_stop_camera_status(self, payload: object) -> None:
        try:
            status = parse_camera_status(payload)
        except ValueError as error:
            message = str(error)
            self.protocol_error.emit(message)
            self.camera_stop_failed.emit(message)
            return
        self.camera_status_changed.emit(status)
        self.camera_stop_completed.emit(status)

    def _handle_validation_report(self, payload: object) -> None:
        try:
            report = parse_validation_report(payload)
        except ValueError as error:
            self.protocol_error.emit(str(error))
            return
        self.validation_report_loaded.emit(report)

    def _handle_manager_catalog(self, payload: object) -> None:
        try:
            entries = parse_manager_catalog(payload)
        except ValueError as error:
            self.management_failed.emit({"message": str(error)})
            return
        self.manager_catalog_loaded.emit(entries)

    def _handle_manager_options(self, payload: object) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("required_ppe"), list):
            self.management_failed.emit({"message": "Core returned invalid PPE options"})
            return
        values = payload["required_ppe"]
        if any(not isinstance(value, str) or not value for value in values):
            self.management_failed.emit({"message": "Core returned invalid PPE options"})
            return
        self.manager_options_loaded.emit(tuple(values))

    def _handle_manager_draft(self, payload: object) -> None:
        try:
            draft = parse_worksite_draft(payload)
        except ValueError as error:
            self.management_failed.emit({"message": str(error)})
            return
        self.manager_draft_loaded.emit(draft)

    def _handle_manager_json(self, payload: object) -> None:
        try:
            document = parse_raw_json_document(payload)
        except ValueError as error:
            self.management_failed.emit({"message": str(error)})
            return
        self.manager_json_loaded.emit(document)

    def _handle_manager_import(self, payload: object) -> None:
        try:
            summary = parse_import_summary(payload)
        except ValueError as error:
            self.management_failed.emit({"message": str(error)})
            return
        self.manager_import_completed.emit(summary)

    def _handle_worker_profile_validated(self, payload: object) -> None:
        try:
            profile = parse_worker_profile(payload)
        except ValueError as error:
            self.management_failed.emit({"message": str(error)})
            return
        self.worker_profile_validated.emit(profile)

    def _handle_worker_profile_saved(self, payload: object) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("profile"), dict):
            self.management_failed.emit({"message": "Core returned an invalid profile result"})
            return
        try:
            profile_payload = dict(payload["profile"])
            if isinstance(payload.get("source"), str):
                profile_payload["source"] = payload["source"]
            profile = parse_worker_profile(profile_payload)
        except ValueError as error:
            self.management_failed.emit({"message": str(error)})
            return
        self.worker_profile_saved.emit(profile)

    def _websocket_url(self) -> str:
        parts = urlsplit(self._base_url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        return urlunsplit((scheme, parts.netloc, "/ws/gui", "", ""))

    def _poll_state(self) -> None:
        if self._state_reply_pending:
            return
        self._state_reply_pending = True
        reply = self._network.get(QNetworkRequest(QUrl(f"{self._base_url}/api/state")))
        reply.finished.connect(lambda: self._finish_state(reply))

    def _finish_state(self, reply: QNetworkReply) -> None:
        self._state_reply_pending = False
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                return
            try:
                payload = json.loads(bytes(reply.readAll()))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if not isinstance(payload, dict):
                return
            state = payload.get("state")
            if state in {"STANDBY", "AUTHORIZATION"}:
                self.state_changed.emit(state)
            camera = payload.get("camera")
            if camera is not None:
                self._handle_camera_status(camera)
        finally:
            reply.deleteLater()

    def _on_message(self, message: str) -> None:
        try:
            outcome: Outcome = parse_outcome(json.loads(message))
        except (json.JSONDecodeError, ValueError) as error:
            self.protocol_error.emit(str(error))
            return
        self.outcome_received.emit(outcome)

    def _on_disconnected(self) -> None:
        self.connection_changed.emit(False)
        self._reconnect.start()

    def _on_socket_error(self, _error: QWebSocket.Error) -> None:
        self.connection_changed.emit(False)
