from __future__ import annotations

import unittest

from signatus_ai.app import events, tracking_events


class DisconnectingSocket:
    def __init__(self) -> None:
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, str]:
        return {"type": "websocket.disconnect"}

    async def send_text(self, _message: str) -> None:
        raise AssertionError("No event should be sent after disconnect")


class AIEventWebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_disconnected_client_does_not_block_ai_shutdown(self) -> None:
        socket = DisconnectingSocket()
        subscriber_count = len(events._subscribers)

        await tracking_events(socket)  # type: ignore[arg-type]

        self.assertTrue(socket.accepted)
        self.assertEqual(len(events._subscribers), subscriber_count)


if __name__ == "__main__":
    unittest.main()
