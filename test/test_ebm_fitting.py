from __future__ import annotations

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

from src.data import PatientLandmarkTable
from src.model.ebm_fitting import fit_ebm_model


def test_ebm_model_fits_weighted_landmarks_and_predicts_risk() -> None:
    table = survival_table()
    config = {"outer_bags": 1, "validation_size": 0, "max_rounds": 5, "n_jobs": 1}

    fitted, metadata = fit_ebm_model(table, table.features, config)

    risk_scores = fitted.predict(table.features)
    assert isinstance(fitted, ExplainableBoostingRegressor)
    assert risk_scores.shape == (len(table.features),)
    assert np.isfinite(risk_scores).all()
    assert tuple(fitted.feature_names_in_) == table.feature_names
    assert metadata["library"] == "interpret"
    assert metadata["config"] == config
    assert metadata["effective_parameters"]["objective"] == "survival_cox"


def survival_table() -> PatientLandmarkTable:
    random = np.random.default_rng(42)
    row_count = 80
    features = random.normal(size=(row_count, 2))
    durations = random.exponential(20.0, size=row_count) + 1.0
    events = random.random(row_count) < 0.6
    return PatientLandmarkTable(
        features=features,
        signed_durations=np.where(events, durations, -durations),
        durations=durations,
        events=events,
        sample_weights=np.resize(np.asarray([0.5, 0.5, 1.0]), row_count),
        feature_names=("blood_a", "blood_b"),
        patient_ids=np.asarray([f"P{index // 2}" for index in range(row_count)]),
        landmark_days=np.arange(row_count),
        prior_non_endpoint_recurrence=np.zeros(row_count, dtype=bool),
        metadata={},
    )
