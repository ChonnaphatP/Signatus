from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import signatus_core.app as core_app
from signatus_contracts import CameraState, CameraStatus
from signatus_core.controller import CameraCommandError, CameraTransitionError
from signatus_core.domain import CoreState


class CoreCameraAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_error_is_reported_without_making_core_unhealthy(self) -> None:
        controller = SimpleNamespace(
            state=CoreState.STANDBY,
            camera_status=CameraStatus(
                state=CameraState.ERROR,
                error="Unable to open configured camera",
            ),
        )
        ai_client = SimpleNamespace(events_connected=True)

        with (
            patch.object(core_app, "controller", controller),
            patch.object(core_app, "ai_client", ai_client),
        ):
            health = await core_app.health()
            camera = await core_app.camera_status()

        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["ai_events_connected"])
        self.assertEqual(health["camera_state"], "ERROR")
        self.assertEqual(camera.state, CameraState.ERROR)

    async def test_state_exposes_camera_state_separately_from_core_state(self) -> None:
        controller = SimpleNamespace(
            state=CoreState.STANDBY,
            camera_status=CameraStatus(state=CameraState.STOPPED),
            selected_worksite=None,
        )
        signals = SimpleNamespace(latest=None)

        with (
            patch.object(core_app, "controller", controller),
            patch.object(core_app, "signals", signals),
        ):
            payload = await core_app.state()

        self.assertEqual(payload["state"], "STANDBY")
        self.assertEqual(payload["camera"], {"state": "STOPPED", "error": None})
        self.assertEqual(payload["camera_state"], "STOPPED")
        self.assertIsNone(payload["camera_error"])

    async def test_camera_routes_forward_valid_commands(self) -> None:
        controller = SimpleNamespace(
            start_camera=AsyncMock(
                return_value=CameraStatus(state=CameraState.STARTING)
            ),
            stop_camera=AsyncMock(
                return_value=CameraStatus(state=CameraState.STOPPING)
            ),
        )

        with patch.object(core_app, "controller", controller):
            started = await core_app.start_camera()
            stopped = await core_app.stop_camera()

        self.assertEqual(started.state, CameraState.STARTING)
        self.assertEqual(stopped.state, CameraState.STOPPING)
        controller.start_camera.assert_awaited_once_with()
        controller.stop_camera.assert_awaited_once_with()

    async def test_invalid_transition_returns_conflict(self) -> None:
        controller = SimpleNamespace(
            start_camera=AsyncMock(
                side_effect=CameraTransitionError(
                    "cannot start camera while it is RUNNING"
                )
            )
        )

        with (
            patch.object(core_app, "controller", controller),
            self.assertRaises(HTTPException) as raised,
        ):
            await core_app.start_camera()

        self.assertEqual(raised.exception.status_code, 409)

    async def test_ai_command_failure_returns_service_unavailable(self) -> None:
        controller = SimpleNamespace(
            stop_camera=AsyncMock(
                side_effect=CameraCommandError("AI camera command unavailable")
            )
        )

        with (
            patch.object(core_app, "controller", controller),
            self.assertRaises(HTTPException) as raised,
        ):
            await core_app.stop_camera()

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
