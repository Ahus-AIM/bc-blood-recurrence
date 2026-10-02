from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from src.model.normal_ranges import NormalRange
from src.script.calculate_recurrence_reference_percentages import AgeBound


def reference_key(test_key: str) -> str:
    """Match name/unit case and whitespace without converting measurement units."""
    return "".join(test_key.casefold().replace("μ", "µ").split())


def reference_ranges_for_test(ranges: Iterable[NormalRange], test_key: str) -> tuple[NormalRange, ...]:
    key = reference_key(test_key)
    return tuple(
        dict.fromkeys(
            interval
            for interval in ranges
            if interval.sex == "female" and reference_key(interval.feature_key) == key and usable_interval(interval)
        )
    )


def usable_interval(interval: NormalRange) -> bool:
    bounds = [bound for bound in (interval.lower, interval.upper) if bound is not None]
    return (
        bool(bounds)
        and bool(np.isfinite(bounds).all())
        and (len(bounds) == 1 or bounds[0] <= bounds[1])
        and (interval.lower is None or interval.lower_operator in {">", ">=", "≥"})
        and (interval.upper is None or interval.upper_operator in {"<", "<=", "≤"})
    )


def reference_age_bounds(label: str) -> list[AgeBound] | None:
    """Translate age labels to calendar-month anniversaries; unknown labels stay unknown."""
    label = "".join(label.casefold().split()).replace("–", "-").replace("≥", ">=").replace("≤", "<=")
    if label == "allages":
        return []
    if label in {"adult", "adultfastingserum", "adultg1"}:
        return [{"months": 216, "operator": ">="}]
    interval = re.fullmatch(r"(\d+)-(\d+)years?", label)
    if interval is not None:
        lower, upper = (int(value) for value in interval.groups())
        if lower > upper:
            return None
        return [{"months": lower * 12, "operator": ">="}, {"months": (upper + 1) * 12, "operator": "<"}]
    bound = re.fullmatch(r"(>=|<=|>|<)(\d+(?:\.\d+)?)(years?|months?)", label)
    if bound is None:
        return None
    operator, number, unit = bound.groups()
    months = float(number) * (12 if unit.startswith("year") else 1)
    return [{"months": int(months), "operator": operator}] if months.is_integer() else None


def reference_age_label(label: str) -> str:
    """Display numeric ages using the same boundaries as the remote classification."""
    bounds = reference_age_bounds(label)
    if bounds is None:
        return label
    if not bounds:
        return "all ages"
    if len(bounds) == 2:
        return f"{bounds[0]['months'] // 12}–{bounds[1]['months'] // 12 - 1} years"
    bound = bounds[0]
    months = bound["months"]
    number, unit = (months // 12, "year") if months % 12 == 0 else (months, "month")
    operator = bound["operator"].replace(">=", "≥").replace("<=", "≤")
    age = f"{operator}{number} {unit}{'' if number == 1 else 's'}"
    normalized = "".join(label.casefold().split())
    qualifiers = {"adultfastingserum": "; fasting serum", "adultg1": "; G1"}
    return age + qualifiers.get(normalized, "")


def reference_interval_label(interval: NormalRange, *, include_sex: bool = False) -> str:
    """Display original numeric and age bounds without losing strict inequalities."""
    lower, upper = interval.lower, interval.upper
    lower_inclusive = interval.lower_operator in {">=", "≥"}
    upper_inclusive = interval.upper_operator in {"<=", "≤"}
    if lower is not None and upper is not None:
        if lower_inclusive and upper_inclusive:
            limits = f"{lower:g}–{upper:g}"
        else:
            left = "≤" if lower_inclusive else "<"
            right = "≤" if upper_inclusive else "<"
            limits = f"{lower:g} {left} x {right} {upper:g}"
    elif lower is not None:
        limits = f"{interval.lower_operator.replace('>=', '≥')}{lower:g}"
    elif upper is not None:
        limits = f"{interval.upper_operator.replace('<=', '≤')}{upper:g}"
    else:
        limits = "Unspecified range"
    group = reference_age_label(interval.age_group)
    if include_sex:
        group = f"{interval.sex.capitalize()}, {group}"
    return f"{limits} ({group})"
