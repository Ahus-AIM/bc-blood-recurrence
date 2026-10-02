from types import SimpleNamespace

import numpy as np
import pytest

from src.model.cause_specific_probability import cumulative_incidence, weighted_breslow
from src.script.brier_model_artifacts import predict_splits


@pytest.mark.parametrize("offset", [-1000.0, 120.0, 1000.0])
def test_centering_removes_large_offsets_and_uses_only_weighted_training_scores(offset: float) -> None:
    features = {
        "train": np.array([[0.0], [1.0], [2.0]]),
        "val": np.array([[4.0]]),
        "test": np.array([[5.0], [6.0]]),
        "empty": np.empty((0, 1)),
    }
    weights = np.array([0.75, 0.75, 1.5])
    model = SimpleNamespace(objective="survival_cox", link_="identity", predict=lambda rows: rows[:, 0] + offset)
    risks, center = predict_splits(model, features, weights)
    assert center == pytest.approx(offset + 1.25)
    for split, rows in features.items():
        np.testing.assert_allclose(risks[split], np.exp(rows[:, 0] - 1.25))
    assert np.average(np.log(risks["train"]), weights=weights) == pytest.approx(0.0, abs=1e-14)


def test_separate_cause_centers_preserve_hazard_jumps_and_probabilities() -> None:
    features = {"train": np.array([[0.1], [0.2], [0.3], [0.4]]), "test": np.array([[0.15], [0.35]])}
    weights = np.array([0.5, 0.5, 1.5, 1.5])
    durations = np.array([0, 10, 10, 180])
    causes = np.array([1, 2, 1, 0])
    predictions = []
    baselines = []
    for cause, offset in ((1, 1000.0), (2, -1000.0)):
        model = SimpleNamespace(
            objective="survival_cox", link_="identity", predict=lambda rows, shift=offset: rows[:, 0] + shift
        )
        centered, _ = predict_splits(model, features, weights)
        raw = {split: np.exp(rows[:, 0]) for split, rows in features.items()}
        fitted = [weighted_breslow(durations, causes == cause, risks["train"], weights) for risks in (raw, centered)]
        np.testing.assert_allclose(
            fitted[0].increments[:, None] * raw["test"],
            fitted[1].increments[:, None] * centered["test"],
            rtol=1e-12,
        )
        predictions.append((raw, centered))
        baselines.append(fitted)
    original, centered = [
        cumulative_incidence(
            baselines[0][index], baselines[1][index], predictions[0][index]["test"], predictions[1][index]["test"]
        )
        for index in range(2)
    ]
    assert original.invalid_hazards == centered.invalid_hazards == ()
    for name in ("target", "competing", "survival"):
        np.testing.assert_allclose(getattr(original, name), getattr(centered, name), rtol=1e-12)


def test_log_link_relative_risks_are_centered_on_the_log_scale() -> None:
    features = {"train": np.array([[0.5], [3.0]]), "test": np.array([[6.0]])}
    model = SimpleNamespace(objective="survival_cox", link_="log", predict=lambda rows: rows[:, 0])
    risks, center = predict_splits(model, features, np.ones(2))
    assert center == pytest.approx(np.log(np.sqrt(1.5)))
    for split, values in features.items():
        np.testing.assert_allclose(risks[split], values[:, 0] / np.sqrt(1.5))


def test_zero_hazard_competitor_uses_unit_risks_and_zero_center() -> None:
    features = {"train": np.zeros((3, 1)), "test": np.empty((0, 1))}
    risks, center = predict_splits(None, features, np.ones(3))
    assert center == 0.0
    np.testing.assert_array_equal(risks["train"], np.ones(3))
    assert risks["test"].size == 0


@pytest.mark.parametrize("score", [-1000.0, 1000.0])
def test_extreme_held_out_risks_still_fail_after_training_centering(score: float) -> None:
    features = {"train": np.array([[0.0]]), "test": np.array([[score]])}
    model = SimpleNamespace(objective="survival_cox", link_="identity", predict=lambda rows: rows[:, 0])
    with pytest.raises(ValueError, match="Centered model relative risks"):
        predict_splits(model, features, np.ones(1))


@pytest.mark.parametrize("risk", [0.0, -1.0, np.nan, np.inf])
def test_invalid_log_link_predictions_fail_before_centering(risk: float) -> None:
    model = SimpleNamespace(objective="survival_cox", link_="log", predict=lambda rows: rows[:, 0])
    with pytest.raises(ValueError, match="Model relative risks"):
        predict_splits(model, {"train": np.array([[risk]])}, np.ones(1))
