import math
import unittest

from signatus_core.domain import AuthorizedWorker
from signatus_core.face_match import cosine_similarity, find_best_match


class FaceMatchTests(unittest.TestCase):
    def test_cosine_similarity_for_same_direction_is_one(self):
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (2.0, 0.0)), 1.0)

    def test_invalid_vectors_do_not_match(self):
        self.assertEqual(cosine_similarity((1.0,), (1.0, 2.0)), -math.inf)
        self.assertEqual(cosine_similarity((0.0, 0.0), (1.0, 0.0)), -math.inf)

    def test_returns_nearest_worker_inside_threshold(self):
        workers = (
            AuthorizedWorker("EMP1", (1.0, 0.0)),
            AuthorizedWorker("EMP2", (0.0, 1.0)),
        )
        result = find_best_match((0.99, 0.01), workers, min_cosine_similarity=0.95)
        self.assertEqual(result.worker_id, "EMP1")

    def test_rejects_best_worker_below_minimum_similarity(self):
        workers = (AuthorizedWorker("EMP1", (1.0, 0.0)),)

        result = find_best_match((0.3, math.sqrt(0.91)), workers, min_cosine_similarity=0.35)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
