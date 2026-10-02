from itertools import product
from random import Random

import numpy as np
import pytest
from scipy.stats import wilcoxon

from src.data.exact_signed_rank import exact_signed_rank
from test.recurrence_statistics_oracles import average_ranks


@pytest.mark.parametrize("seed", range(12))
def test_exact_distribution_matches_enumerating_every_sign_assignment(seed: int) -> None:
    rng = Random(seed)
    differences = [float(rng.randint(-3, 3)) for _ in range(12)]
    nonzero = [value for value in differences if value != 0]
    ranks = average_ranks([abs(value) for value in nonzero])
    positive = sum(rank for value, rank in zip(nonzero, ranks, strict=True) if value > 0)
    total = sum(ranks)
    observed = min(positive, total - positive)
    extreme = 0
    for signs in product((False, True), repeat=len(nonzero)):
        permuted = sum(rank for rank, sign in zip(ranks, signs, strict=True) if sign)
        extreme += min(permuted, total - permuted) <= observed
    assert exact_signed_rank(np.array(differences)) == (observed, extreme / 2**len(nonzero))


def test_tied_ranks_do_not_use_the_untied_exact_distribution() -> None:
    # With three equal absolute changes, every sign assignment is at least
    # as extreme as two increases and one decrease: the exact p-value is one.
    assert exact_signed_rank(np.array([1.0, 1.0, -1.0])) == (2.0, 1.0)


def test_large_untied_sample_matches_scipy_exact_distribution() -> None:
    differences = np.arange(1, 115, dtype=float)
    differences[::3] *= -1
    expected = wilcoxon(differences, method="exact")
    statistic, p_value = exact_signed_rank(differences)
    assert statistic == expected.statistic
    assert p_value == pytest.approx(expected.pvalue, rel=1e-13)


def test_exact_result_is_invariant_to_order_and_global_sign_reversal() -> None:
    differences = np.array([0.0, 1.0, -2.0, 2.0, 3.0, -3.0, 0.0, 4.0] * 8)
    expected = exact_signed_rank(differences)
    assert exact_signed_rank(-differences) == expected
    assert exact_signed_rank(differences[::-1]) == expected


@pytest.mark.parametrize("differences", [[], [0.0], [0.0] * 20])
def test_empty_or_all_zero_changes_have_unit_p_value(differences: list[float]) -> None:
    assert exact_signed_rank(np.array(differences, dtype=float)) == (0.0, 1.0)


@pytest.mark.parametrize("differences", [np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2))])
def test_invalid_inputs_raise_instead_of_falling_back_to_approximation(differences: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite one-dimensional"):
        exact_signed_rank(differences)
