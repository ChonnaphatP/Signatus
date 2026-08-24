from __future__ import annotations

import math
from collections.abc import Sequence

from .domain import AuthorizedWorker


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return -math.inf

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -math.inf
    return dot / (left_norm * right_norm)


def find_best_match(
    candidate: Sequence[float],
    workers: Sequence[AuthorizedWorker],
    min_cosine_similarity: float,
) -> AuthorizedWorker | None:
    best_worker: AuthorizedWorker | None = None
    best_similarity = -math.inf

    for worker in workers:
        similarity = cosine_similarity(candidate, worker.embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_worker = worker

    if best_worker is None or best_similarity < min_cosine_similarity:
        return None
    return best_worker
