"""Shared conversions for InterpretML explanation data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def float_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float)


def optional_float_array(values: Iterable[object]) -> np.ndarray | None:
    try:
        return np.asarray(list(values), dtype=float)
    except (TypeError, ValueError):
        return None


def sequence_value(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return tuple(value) if isinstance(value, Iterable) else (value,)
