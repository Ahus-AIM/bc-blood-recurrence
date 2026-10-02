"""Exact conditional Wilcoxon inference with tied ranks and omitted zero changes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


def exact_signed_rank(differences: NDArray[np.float64]) -> tuple[float, float]:
    """Return W and its two-sided exact p-value, conditional on absolute changes.

    Zero differences are omitted before assigning average ranks. Each sign
    assignment is equally likely under the symmetric-difference null. Integer
    subset-sum counts give its exact distribution, including ties, without
    enumerating all 2**n assignments. Counts remain integers; the p-value is
    returned as a float.
    """
    if differences.ndim != 1 or not np.isfinite(differences).all():
        raise ValueError("Signed-rank differences must be a finite one-dimensional array.")
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0, 1.0
    # Average ranks can be half integers; doubling preserves them exactly.
    weights = (2 * rankdata(np.abs(nonzero), method="average")).astype(np.int64)
    positive = int(weights[nonzero > 0].sum())
    statistic = min(positive, int(weights.sum()) - positive)
    divisor = int(np.gcd.reduce(weights))
    tail_count = lower_tail_count(weights // divisor, statistic // divisor)
    assignments = 1 << len(nonzero)
    p_value = min(assignments, 2 * tail_count) / assignments
    return statistic / 2, p_value


def lower_tail_count(weights: NDArray[np.int64], limit: int) -> int:
    """Count subsets with rank sum ≤ limit using arbitrary-precision integers."""
    counts = np.zeros(limit + 1, dtype=object)
    counts[0] = 1
    for weight in weights:
        if weight <= limit:
            # Each subset either excludes this rank or includes it once.
            counts[weight:] = counts[weight:] + counts[:-weight]
    return int(counts.sum())
