from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from signatus_ai.settings import AISettings


class AISettingsTests(unittest.TestCase):
    def test_approved_person_class_and_association_are_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = AISettings.from_environment()

        self.assertEqual(settings.person_class, "Person")
        self.assertEqual(settings.ppe_association, "single_person_frame")
        self.assertEqual(
            settings.face_detector_model_path.name,
            "face_detection_yunet_2023mar.onnx",
        )
        self.assertEqual(
            settings.face_recognizer_model_path.name,
            "face_recognition_sface_2021dec.onnx",
        )
        self.assertEqual(settings.face_detector_score_threshold, 0.9)
        self.assertEqual(settings.face_detector_nms_threshold, 0.3)
        self.assertEqual(settings.face_detector_top_k, 5000)
        self.assertTrue(settings.frame_shm_enabled)
        self.assertEqual(settings.frame_shm_name, "signatus_camera_v1")
        self.assertEqual(settings.preview_max_frame_bytes, 1920 * 1080 * 3)

    def test_preview_shared_memory_can_be_configured_or_disabled(self) -> None:
        environment = {
            "SIGNATUS_FRAME_SHM_ENABLED": "off",
            "SIGNATUS_FRAME_SHM_NAME": " checkpoint_camera ",
            "SIGNATUS_PREVIEW_MAX_FRAME_BYTES": "123456",
        }

        with patch.dict(os.environ, environment, clear=True):
            settings = AISettings.from_environment()

        self.assertFalse(settings.frame_shm_enabled)
        self.assertEqual(settings.frame_shm_name, "checkpoint_camera")
        self.assertEqual(settings.preview_max_frame_bytes, 123456)


if __name__ == "__main__":
    unittest.main()
