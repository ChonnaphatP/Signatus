from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CoreSettings:
    ai_base_url: str
    ai_events_url: str
    worksite_dir: Path
    worker_profile_dir: Path
    face_match_min_cosine_similarity: float

    @classmethod
    def from_environment(cls) -> CoreSettings:
        return cls(
            ai_base_url=os.getenv("SIGNATUS_AI_BASE_URL", "http://127.0.0.1:8001"),
            ai_events_url=os.getenv("SIGNATUS_AI_EVENTS_URL", "ws://127.0.0.1:8001/ws/events"),
            worksite_dir=Path(os.getenv("SIGNATUS_WORKSITE_DIR", "./config/worksites")),
            worker_profile_dir=Path(
                os.getenv("SIGNATUS_WORKER_PROFILE_DIR", "./config/worker_profiles")
            ),
            face_match_min_cosine_similarity=float(
                os.getenv("SIGNATUS_FACE_MATCH_MIN_COSINE_SIMILARITY", "0.35")
            ),
        )
