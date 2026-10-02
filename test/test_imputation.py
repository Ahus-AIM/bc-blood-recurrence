from __future__ import annotations

import numpy as np
import pytest

from src.data import PatientLandmarkTable
from src.model.imputation import ImputationConfig, fit_feature_imputer


def test_mice_imputer_transforms_missing_cells_and_records_metadata() -> None:
    features = np.asarray(
        [
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    config = ImputationConfig(enabled=True, method="mice", max_iter=2, random_state=7)

    fitted = fit_feature_imputer(features, ("A", "B"), config)
    result = fitted.transform(np.asarray([[np.nan, 40.0], [5.0, np.nan]], dtype=np.float32))

    assert result.imputed_mask.tolist() == [[True, False], [False, True]]
    assert result.clipped_mask.tolist() == [[False, False], [False, False]]
    assert np.isfinite(result.features).all()
    assert fitted.metadata["training_missing_cell_count"] == 1
    assert fitted.metadata["training_missing_counts_by_feature"] == {"B": 1}
    assert fitted.metadata["tol"] == 0.001
    assert fitted.metadata["completed_iterations"] >= 1


def test_mice_imputer_clips_only_negative_imputed_nonnegative_features() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 10.0],
            [3.0, 20.0],
        ],
        dtype=np.float32,
    )
    config = ImputationConfig(enabled=True, method="mice", max_iter=5, random_state=0)
    fitted = fit_feature_imputer(
        features,
        ("A|||u", "A|||u change"),
        config,
        nonnegative_feature_names=("A|||u",),
    )

    result = fitted.transform(np.asarray([[np.nan, -20.0], [np.nan, 30.0]], dtype=np.float32))

    assert result.features[0, 0] == 0.0
    assert result.features[0, 1] == -20.0
    assert result.features[1, 0] > 0.0
    assert result.clipped_mask.tolist() == [[True, False], [False, False]]
    assert fitted.metadata["nonnegative_feature_names"] == ("A|||u",)


def test_mice_imputer_rejects_all_missing_training_feature() -> None:
    features = np.asarray(
        [
            [1.0, np.nan],
            [2.0, np.nan],
        ],
        dtype=np.float32,
    )
    config = ImputationConfig(enabled=True, method="mice", max_iter=2, random_state=7)

    with pytest.raises(ValueError, match="all training values missing: B"):
        fit_feature_imputer(features, ("A", "B"), config)


def test_locf_imputer_carries_previous_patient_landmark_and_uses_training_median() -> None:
    features = np.asarray(
        [
            [1.0, np.nan, 70.0],
            [np.nan, 4.0, 70.0],
            [np.nan, np.nan, 70.0],
            [10.0, 8.0, 60.0],
        ],
        dtype=np.float32,
    )
    table = landmark_table(
        features,
        ("A|||u", "B|||u", "age_at_diagnosis"),
        patient_ids=("P0", "P0", "P0", "P1"),
        landmark_days=(10, 20, 30, 10),
    )
    config = ImputationConfig(enabled=True, method="locf", max_iter=0, random_state=42, tol=0.0)

    fitted = fit_feature_imputer(
        features,
        table.feature_names,
        config,
        nonnegative_feature_names=("A|||u", "B|||u"),
        landmark_table=table,
    )
    result = fitted.transform_table(table)

    np.testing.assert_allclose(
        result.features,
        np.asarray(
            [
                [1.0, 4.0, 70.0],
                [1.0, 4.0, 70.0],
                [1.0, 4.0, 70.0],
                [10.0, 8.0, 60.0],
            ],
            dtype=np.float32,
        ),
    )
    assert result.imputed_mask.tolist() == [
        [False, True, False],
        [True, False, False],
        [True, True, False],
        [False, False, False],
    ]
    assert result.carried_forward_mask is not None
    assert result.carried_forward_mask.tolist() == [
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [False, False, False],
    ]
    assert result.median_fallback_mask is not None
    assert result.median_fallback_mask.tolist() == [
        [False, True, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    assert fitted.metadata["method"] == "locf"
    assert fitted.metadata["fallback_method"] == "median"
    assert fitted.metadata["training_median_by_locf_feature"] == {"A|||u": 1.0, "B|||u": 4.0}
    assert fitted.metadata["training_remaining_missing_after_locf_cell_count"] == 1


def test_locf_imputer_uses_reference_history_for_bootstrap_rows() -> None:
    reference = landmark_table(
        np.asarray(
            [
                [1.0, np.nan],
                [np.nan, 4.0],
                [10.0, 40.0],
            ],
            dtype=np.float32,
        ),
        ("A|||u", "B|||u"),
        patient_ids=("P0", "P0", "P1"),
        landmark_days=(10, 20, 10),
    )
    bootstrap = landmark_table(
        np.asarray([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32),
        reference.feature_names,
        patient_ids=("P0", "P0"),
        landmark_days=(30, 30),
    )
    config = ImputationConfig(enabled=True, method="locf", max_iter=10, random_state=42)

    fitted = fit_feature_imputer(
        reference.features,
        reference.feature_names,
        config,
        nonnegative_feature_names=reference.feature_names,
        landmark_table=reference,
    )
    result = fitted.transform_table(bootstrap, reference_table=reference)

    np.testing.assert_allclose(result.features, np.asarray([[1.0, 4.0], [1.0, 4.0]], dtype=np.float32))
    assert result.carried_forward_mask is not None
    assert result.carried_forward_mask.tolist() == [[True, True], [True, True]]
    assert result.median_fallback_mask is not None
    assert result.median_fallback_mask.tolist() == [[False, False], [False, False]]


def test_locf_median_fallback_fills_remaining_missing_non_blood_feature() -> None:
    features = np.asarray([[1.0, np.nan], [2.0, 70.0]], dtype=np.float32)
    table = landmark_table(
        features,
        ("A|||u", "age_at_diagnosis"),
        patient_ids=("P0", "P1"),
        landmark_days=(10, 10),
    )
    config = ImputationConfig(enabled=True, method="locf", max_iter=10, random_state=42)
    fitted = fit_feature_imputer(
        features,
        table.feature_names,
        config,
        nonnegative_feature_names=("A|||u",),
        landmark_table=table,
    )

    result = fitted.transform_table(table)

    np.testing.assert_allclose(result.features, np.asarray([[1.0, 70.0], [2.0, 70.0]], dtype=np.float32))
    assert result.median_fallback_mask is not None
    assert result.median_fallback_mask.tolist() == [[False, True], [False, False]]
    assert fitted.metadata["training_median_by_fallback_feature"]["age_at_diagnosis"] == 70.0


def test_locf_transform_requires_landmark_context() -> None:
    features = np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    table = landmark_table(features, ("A|||u", "B|||u"), patient_ids=("P0", "P1"), landmark_days=(10, 20))
    config = ImputationConfig(enabled=True, method="locf", max_iter=10, random_state=42)
    fitted = fit_feature_imputer(
        features,
        ("A|||u", "B|||u"),
        config,
        nonnegative_feature_names=("A|||u", "B|||u"),
        landmark_table=table,
    )

    with pytest.raises(ValueError, match="requires patient ids and landmark days"):
        fitted.transform(features)


def test_imputation_config_reads_and_validates_tolerance() -> None:
    config = ImputationConfig.from_mapping(
        {"enabled": True, "method": "mice", "max_iter": 5, "random_state": 7, "tol": 0.01}
    )

    assert config.tol == 0.01

    with pytest.raises(ValueError, match="imputation.tol"):
        ImputationConfig.from_mapping({"enabled": True, "method": "mice", "tol": 0.0})


def test_imputation_config_accepts_locf_and_ignores_mice_iteration_settings() -> None:
    config = ImputationConfig.from_mapping({"enabled": True, "method": "locf", "max_iter": 0, "tol": 0.0})

    assert config.method == "locf"


def test_bounded_locf_expires_old_history_before_median_fallback() -> None:
    features = np.asarray(
        [
            [1.0],
            [np.nan],
            [np.nan],
        ],
        dtype=np.float32,
    )
    table = landmark_table(features, ("A|||u",), patient_ids=("P0", "P0", "P0"), landmark_days=(10, 20, 500))
    config = ImputationConfig.from_mapping(
        {"enabled": True, "method": "locf", "locf_max_age_days": 365, "locf_fallback_method": "median"}
    )

    fitted = fit_feature_imputer(
        features,
        table.feature_names,
        config,
        nonnegative_feature_names=table.feature_names,
        landmark_table=table,
    )
    result = fitted.transform_table(table)

    np.testing.assert_allclose(result.features, np.asarray([[1.0], [1.0], [1.0]], dtype=np.float32))
    assert result.carried_forward_mask is not None
    assert result.carried_forward_mask.tolist() == [[False], [True], [False]]
    assert result.locf_expired_mask is not None
    assert result.locf_expired_mask.tolist() == [[False], [False], [True]]
    assert result.fallback_mask is not None
    assert result.fallback_mask.tolist() == [[False], [False], [True]]


def test_locf_mice_fallback_fills_values_remaining_after_carry_forward() -> None:
    features = np.asarray(
        [
            [1.0, 10.0],
            [np.nan, 20.0],
            [3.0, np.nan],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )
    table = landmark_table(
        features,
        ("A|||u", "B|||u"),
        patient_ids=("P0", "P1", "P2", "P3"),
        landmark_days=(10, 10, 10, 10),
    )
    config = ImputationConfig.from_mapping(
        {"enabled": True, "method": "locf", "locf_fallback_method": "mice", "max_iter": 5, "random_state": 3}
    )

    fitted = fit_feature_imputer(
        features,
        table.feature_names,
        config,
        nonnegative_feature_names=table.feature_names,
        landmark_table=table,
    )
    result = fitted.transform_table(table)

    assert np.isfinite(result.features).all()
    assert result.mice_fallback_mask is not None
    assert result.mice_fallback_mask.tolist() == [[False, False], [True, False], [False, True], [False, False]]
    assert result.fallback_method == "mice"
    assert fitted.metadata["fallback_method"] == "mice"
    assert fitted.metadata["max_iter_reached"] is False


def test_imputation_config_validates_locf_controls() -> None:
    with pytest.raises(ValueError, match="locf_max_age_days"):
        ImputationConfig.from_mapping({"enabled": True, "method": "locf", "locf_max_age_days": 0})

    with pytest.raises(ValueError, match="locf_fallback_method"):
        ImputationConfig.from_mapping({"enabled": True, "method": "locf", "locf_fallback_method": "mean"})

    with pytest.raises(ValueError, match="max_iter"):
        ImputationConfig.from_mapping(
            {"enabled": True, "method": "locf", "locf_fallback_method": "mice", "max_iter": 0}
        )


def test_disabled_imputer_keeps_features_unchanged() -> None:
    features = np.asarray([[1.0, np.nan]], dtype=np.float32)
    config = ImputationConfig(enabled=False, method="mice", max_iter=10, random_state=42)

    fitted = fit_feature_imputer(features, ("A", "B"), config)
    result = fitted.transform(features)

    assert result.features is features
    assert result.imputed_mask.tolist() == [[False, False]]
    assert result.clipped_mask.tolist() == [[False, False]]
    assert fitted.metadata == {"enabled": False}


def landmark_table(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    patient_ids: tuple[str, ...],
    landmark_days: tuple[int, ...],
) -> PatientLandmarkTable:
    durations = np.full(len(patient_ids), 10, dtype=np.int32)
    events = np.zeros(len(patient_ids), dtype=bool)
    return PatientLandmarkTable(
        features=features,
        signed_durations=-durations.astype(np.float64),
        durations=durations,
        events=events,
        sample_weights=None,
        feature_names=feature_names,
        patient_ids=np.asarray(patient_ids, dtype=object),
        landmark_days=np.asarray(landmark_days, dtype=np.int32),
        prior_non_endpoint_recurrence=np.zeros(len(patient_ids), dtype=bool),
        metadata={},
    )
