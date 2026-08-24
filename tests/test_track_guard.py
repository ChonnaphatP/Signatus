import unittest

from signatus_core.track_guard import InMemoryTrackGuard


class TrackGuardTests(unittest.TestCase):
    def test_face_failure_retry_and_handle_limit(self):
        guard = InMemoryTrackGuard(retry_cooldown_seconds=1.0, max_face_failures=3)
        guard.observe(7, 10.0)
        self.assertTrue(guard.should_trigger(7, 10.0))

        first = guard.record_face_failure(7, 10.0)
        self.assertEqual(first.attempt, 1)
        self.assertTrue(first.retry_allowed)
        self.assertFalse(guard.should_trigger(7, 10.9))
        self.assertTrue(guard.should_trigger(7, 11.0))

        guard.record_face_failure(7, 11.0)
        third = guard.record_face_failure(7, 12.0)
        self.assertEqual(third.attempt, 3)
        self.assertFalse(third.retry_allowed)
        self.assertFalse(guard.should_trigger(7, 99.0))

    def test_track_lost_forgets_all_state(self):
        guard = InMemoryTrackGuard()
        guard.observe(3, 1.0)
        guard.mark_handled(3)
        guard.forget(3)
        guard.observe(3, 2.0)
        self.assertTrue(guard.should_trigger(3, 2.0))


if __name__ == "__main__":
    unittest.main()
