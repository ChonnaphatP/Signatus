from __future__ import annotations

from .domain import PPEClassRule, PPEEvaluation

PPE_CLASS_MAP: dict[str, PPEClassRule] = {
    "helmet": PPEClassRule(frozenset({"helmet"}), frozenset({"no_helmet"})),
    "gloves": PPEClassRule(frozenset({"gloves"}), frozenset({"no_gloves"})),
    "vest": PPEClassRule(frozenset({"vest"}), frozenset()),
    "boots": PPEClassRule(frozenset({"boots"}), frozenset({"no_boots"})),
    "goggles": PPEClassRule(frozenset({"goggles"}), frozenset({"no_goggle"})),
}
FULL_ABSENCE_CLASS = "none"


def normalize_class_name(value: str) -> str:
    return value.strip().casefold()


def evaluate_ppe(
    required_ppe: tuple[str, ...],
    detected_classes: tuple[str, ...],
    policy: dict[str, PPEClassRule],
) -> PPEEvaluation:
    detected = {normalize_class_name(name) for name in detected_classes}
    if FULL_ABSENCE_CLASS in detected:
        missing = tuple(required_ppe)
        return PPEEvaluation(compliant=not missing, missing_ppe=missing)

    missing: list[str] = []

    for required_item in required_ppe:
        key = normalize_class_name(required_item)
        rule = policy.get(key)
        if rule is None:
            raise KeyError(f"No PPE class policy exists for {required_item!r}")

        negative_hit = bool(rule.negative_classes & detected)
        positive_hit = bool(rule.positive_classes & detected)
        if negative_hit or not positive_hit:
            missing.append(required_item)

    return PPEEvaluation(compliant=not missing, missing_ppe=tuple(missing))
