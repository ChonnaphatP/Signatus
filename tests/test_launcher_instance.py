from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from signatus_launcher.instance import SingleInstanceError, SingleInstanceLock


class SingleInstanceLockTests(unittest.TestCase):
    def test_second_launcher_is_rejected_until_owner_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.lock"
            first = SingleInstanceLock(path, launch_id="launch-one")
            second = SingleInstanceLock(path, launch_id="launch-two")
            first.acquire()
            try:
                with self.assertRaisesRegex(
                    SingleInstanceError,
                    "launch_id=launch-one",
                ):
                    second.acquire()
            finally:
                first.release()

            second.acquire()
            second.release()

    def test_stale_lock_file_does_not_claim_runtime_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.lock"
            path.write_text(
                json.dumps({"launch_id": "old", "launcher_pid": 123}),
                encoding="utf-8",
            )
            lock = SingleInstanceLock(path, launch_id="current")
            lock.acquire()
            try:
                owner = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(owner["launch_id"], "current")
                self.assertTrue(lock.acquired)
            finally:
                lock.release()

        self.assertFalse(lock.acquired)


if __name__ == "__main__":
    unittest.main()
