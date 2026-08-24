import unittest

from signatus_core.ppe import FULL_ABSENCE_CLASS, PPE_CLASS_MAP, evaluate_ppe


class PPETests(unittest.TestCase):
    def test_all_required_positive_classes_pass(self):
        result = evaluate_ppe(
            ("helmet", "gloves"),
            ("helmet", "gloves"),
            PPE_CLASS_MAP,
        )
        self.assertTrue(result.compliant)
        self.assertEqual(result.missing_ppe, ())

    def test_negative_class_wins_over_positive_class(self):
        result = evaluate_ppe(
            ("helmet", "gloves"),
            ("helmet", "gloves", "no_gloves"),
            PPE_CLASS_MAP,
        )
        self.assertFalse(result.compliant)
        self.assertEqual(result.missing_ppe, ("gloves",))

    def test_no_positive_or_negative_detection_fails_closed(self):
        result = evaluate_ppe(
            ("helmet", "gloves"),
            ("helmet",),
            PPE_CLASS_MAP,
        )
        self.assertFalse(result.compliant)
        self.assertEqual(result.missing_ppe, ("gloves",))

    def test_missing_vest_positive_fails_closed_without_negative_class(self):
        result = evaluate_ppe(("vest",), (), PPE_CLASS_MAP)

        self.assertFalse(result.compliant)
        self.assertEqual(result.missing_ppe, ("vest",))

    def test_goggles_negative_uses_exact_singular_model_class(self):
        result = evaluate_ppe(
            ("goggles",),
            ("goggles", "no_goggle"),
            PPE_CLASS_MAP,
        )

        self.assertFalse(result.compliant)
        self.assertEqual(result.missing_ppe, ("goggles",))

    def test_full_absence_class_overrides_coexisting_positives(self):
        required = ("helmet", "gloves", "vest", "boots", "goggles")
        detected = (*required, FULL_ABSENCE_CLASS)

        result = evaluate_ppe(required, detected, PPE_CLASS_MAP)

        self.assertFalse(result.compliant)
        self.assertEqual(result.missing_ppe, required)


if __name__ == "__main__":
    unittest.main()
