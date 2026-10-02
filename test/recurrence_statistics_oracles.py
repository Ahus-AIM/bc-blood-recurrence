"""Independent, small-sample arithmetic for auditing recurrence statistics."""

from __future__ import annotations

from collections.abc import Sequence
from math import erfc, sqrt


def average_ranks(values: Sequence[float]) -> list[float]:
    """Assign tied average ranks without scipy or pandas."""
    ordered = sorted(values)
    return [sum(i + 1 for i, other in enumerate(ordered) if other == value) / ordered.count(value)
            for value in values]


def signed_rank_reference(differences: Sequence[float], *, exact: bool) -> tuple[float, float]:
    """Use the sign-flip distribution, or its tie-adjusted normal approximation."""
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 0.0, 1.0
    ranks = average_ranks([abs(value) for value in nonzero])
    positive = sum(rank for value, rank in zip(nonzero, ranks, strict=True) if value > 0)
    statistic = min(positive, sum(ranks) - positive)
    if not exact:
        variance = sum(rank * rank for rank in ranks) / 4
        z = abs(positive - sum(ranks) / 2) / sqrt(variance)
        return statistic, erfc(z / sqrt(2))
    # Twice the midranks are integers. Count all 2**n sign assignments by
    # dynamic programming, including assignments with the observed statistic.
    counts = {0: 1}
    for rank in ranks:
        increment = int(2 * rank)
        updated = counts.copy()
        for total, count in counts.items():
            updated[total + increment] = updated.get(total + increment, 0) + count
        counts = updated
    tail = sum(count for total, count in counts.items() if total <= int(2 * statistic))
    return statistic, min(1.0, 2 * tail / 2 ** len(nonzero))


def bh_reference(values: Sequence[float]) -> list[float]:
    """Apply the BH step-up adjustment, restoring the original hypothesis order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    previous = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        previous = min(previous, len(values) * values[index] / rank)
        adjusted[index] = previous
    return adjusted


def linear_quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    """Interpolate the three order-statistic quantiles without numpy or pandas."""
    if not values:
        return (float("nan"),) * 3
    ordered = sorted(values)
    result = []
    for fraction in (0.25, 0.5, 0.75):
        position = fraction * (len(ordered) - 1)
        index = int(position)
        upper = min(index + 1, len(ordered) - 1)
        result.append(ordered[index] + (position - index) * (ordered[upper] - ordered[index]))
    return result[0], result[1], result[2]
