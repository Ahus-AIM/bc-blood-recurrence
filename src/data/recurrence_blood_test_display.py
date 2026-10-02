from __future__ import annotations

import re


def window_display_label(label: str) -> str:
    """Abbreviate month/week ranges and read them from farther to nearer recurrence."""
    label = re.sub(r"\bmonths?\b", "mo", label, flags=re.IGNORECASE)
    label = re.sub(r"\bweeks?\b", "wk", label, flags=re.IGNORECASE)
    bounds = re.fullmatch(r"(\d+(?:\.\d+)?)(?: (mo|wk))?–(<\d+(?:\.\d+)?) (mo|wk)", label)
    if bounds is None:
        return label
    nearer, nearer_unit, farther, farther_unit = bounds.groups()
    if nearer_unit is None or nearer_unit == farther_unit:
        return f"{farther}–{nearer} {farther_unit}"
    return f"{farther} {farther_unit}–{nearer} {nearer_unit}"
