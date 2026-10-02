from __future__ import annotations

import numpy as np
import pytest
from interpret.utils._native import Booster, Native
from numpy.typing import NDArray

from src.data.survival_targets import signed_survival_durations


@pytest.mark.parametrize(
    ("event_day", "censor_day", "expected_risk_count"),
    [(0, 0, 2), (0, 1, 2), (1, 0, 1), (30, 29, 1), (30, 30, 2), (30, 31, 2),
     (180, 179, 1), (180, 180, 2), (180, 181, 2)],
)
def test_native_cox_includes_same_day_censoring_regardless_of_row_order(
    event_day: int,
    censor_day: int,
    expected_risk_count: int,
) -> None:
    targets = signed_survival_durations(np.array([event_day, censor_day]), np.array([True, False]))

    # At equal zero scores, the sole event contributes log(number at risk).
    # InterpretML reports the mean over both validation rows.
    expected_loss = np.log(expected_risk_count) / 2
    assert native_zero_score_loss(targets) == pytest.approx(expected_loss, abs=1e-12)
    assert native_zero_score_loss(targets[::-1]) == pytest.approx(expected_loss, abs=1e-12)


def native_zero_score_loss(validation_targets: NDArray[np.float64]) -> float:
    """Evaluate the installed Cox objective at fixed scores, without fitting or random bags."""
    native = Native.get_native_singleton()
    targets = np.array([1.0, 2.0, *validation_targets], dtype=np.float64)
    feature = np.ones(len(targets), dtype=np.int64)
    byte_count = native.measure_dataset_header(1, 0, 1)
    byte_count += native.measure_feature(3, True, True, False, feature)
    byte_count += native.measure_regression_target(targets)
    dataset = np.empty(byte_count, dtype=np.ubyte)
    native.fill_dataset_header(1, 0, 1, dataset)
    native.fill_feature(3, True, True, False, feature, dataset)
    native.fill_regression_target(targets, dataset)
    bag = np.array([1, 1, -1, -1], dtype=np.int8)

    with Booster(
        dataset=dataset,
        intercept=np.zeros(1),
        bag=bag,
        init_scores=None,
        term_features=[(0,)],
        n_inner_bags=0,
        rng=None,
        create_booster_flags=0,
        objective="survival_cox",
        acceleration=0,
        experimental_params=None,
    ) as booster:
        booster.set_term_update(0, np.zeros(3))
        return float(booster.apply_term_update())
