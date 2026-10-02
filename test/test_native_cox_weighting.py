from __future__ import annotations

import numpy as np
import pytest
from interpret.utils._native import Booster, Native

from src.data.sample_weighting import patient_uniform_weights


def native_cox_booster(
    targets: np.ndarray, weights: np.ndarray, feature: np.ndarray,
    *, bag: np.ndarray | None = None, scores: np.ndarray | None = None,
) -> Booster:
    """Build a deterministic native probe with explicit row weights and scores."""
    native = Native.get_native_singleton()
    bins = int(feature.max()) + 2
    size = native.measure_dataset_header(1, 1, 1)
    size += native.measure_feature(bins, True, True, False, feature)
    size += native.measure_weight(weights) + native.measure_regression_target(targets)
    dataset = np.empty(size, dtype=np.ubyte)
    native.fill_dataset_header(1, 1, 1, dataset)
    native.fill_feature(bins, True, True, False, feature, dataset)
    native.fill_weight(weights, dataset)
    native.fill_regression_target(targets, dataset)
    return Booster(
        dataset=dataset, intercept=np.zeros(1), bag=bag, init_scores=scores,
        term_features=[(0,)], n_inner_bags=0, rng=None, create_booster_flags=0,
        objective="survival_cox", acceleration=0, experimental_params=None,
    )


def native_intercept_update(targets: np.ndarray, weights: np.ndarray) -> float:
    """Request one intercept-only Newton step from the installed Cox objective."""
    with native_cox_booster(targets, weights, np.ones(len(targets), dtype=np.int64)) as booster:
        booster.generate_term_update(
            rng=None, term_idx=-1, term_boost_flags=0, learning_rate=1.0,
            min_samples_leaf=0, min_hessian=0.0, reg_alpha=0.0, reg_lambda=0.0,
            max_delta_step=0.0, min_cat_samples=0, cat_smooth=0.0,
            max_cat_threshold=32, cat_include=1.0, max_leaves=1, monotone_constraints=None,
        )
        return float(booster.get_term_update().item())


def test_native_cox_uniform_weights_have_zero_intercept_update() -> None:
    assert native_intercept_update(np.array([1.0, -2.0]), np.ones(2)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Open F6: native Cox gradients mishandle unequal sample weights.")
@pytest.mark.parametrize("weights", [np.array([2.0, 1.0]), np.array([1.0, 2.0])])
def test_native_cox_unequal_weights_must_preserve_zero_intercept_gradient(weights: np.ndarray) -> None:
    # A common shift cancels from every Cox partial-likelihood term, including weighted terms.
    assert native_intercept_update(np.array([1.0, -2.0]), weights) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="Open F6: patient-uniform landmark weights trigger spurious Cox updates.",
)
def test_native_cox_patient_uniform_weights_must_preserve_zero_intercept_gradient() -> None:
    weights = patient_uniform_weights(np.array(["A", "B", "B"]))
    assert native_intercept_update(np.array([1.0, -2.0, -3.0]), weights) == pytest.approx(0.0, abs=1e-12)


def native_validation_loss(scores: np.ndarray) -> float:
    targets = np.tile(np.array([1.0, 2.0, -3.0]), 2)
    weights = np.tile(np.array([2.0, 1.0, 1.0]), 2)
    with native_cox_booster(
        targets, weights, np.ones(6, dtype=np.int64),
        bag=np.array([1, 1, 1, -1, -1, -1], dtype=np.int8), scores=np.concatenate((np.zeros(3), scores)),
    ) as booster:
        booster.set_term_update(0, np.zeros(3))
        return float(booster.apply_term_update())


def test_native_weighted_validation_loss_is_constant_under_common_score_shifts() -> None:
    assert native_validation_loss(np.full(3, 100.0)) == pytest.approx(native_validation_loss(np.zeros(3)), abs=1e-12)


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="Open F6: native weighted gradients disagree with the reported loss beyond a common offset.",
)
def test_native_weighted_gradient_matches_loss_for_a_zero_sum_score_change() -> None:
    targets = np.array([1.0, 2.0, -3.0])
    weights = np.array([2.0, 1.0, 1.0])
    with native_cox_booster(targets, weights, np.arange(1, 4, dtype=np.int64)) as booster:
        booster.generate_term_update(
            rng=None, term_idx=0,
            term_boost_flags=Native.TermBoostFlags_GradientSums | Native.TermBoostFlags_RandomSplits,
            learning_rate=1.0, min_samples_leaf=0, min_hessian=0.0, reg_alpha=0.0, reg_lambda=0.0,
            max_delta_step=0.0, min_cat_samples=0, cat_smooth=0.0,
            max_cat_threshold=32, cat_include=1.0, max_leaves=5, monotone_constraints=None,
        )
        gradient = booster.get_term_update()[1:4]
    direction = np.array([1.0, -1.0, 0.0])
    epsilon = 1e-5
    finite_difference = (
        native_validation_loss(epsilon * direction) - native_validation_loss(-epsilon * direction)
    ) / (2 * epsilon)
    assert np.dot(gradient, direction) / weights.sum() == pytest.approx(finite_difference, abs=1e-9)
