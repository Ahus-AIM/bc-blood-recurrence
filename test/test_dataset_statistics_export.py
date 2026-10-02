from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.cohort_comparison import COMPARISON_VERSION
from src.data.cohort_selection import COHORT_SELECTION_VERSION
from src.data.dataset_statistics import DatasetStatisticsSummary
from src.data.dataset_statistics_export import (
    dataset_statistics_payload,
    dataset_statistics_updates_from_payload,
    included_patient_ids_from_payload,
    load_dataset_statistics_payload,
    recurrence_incidence_from_payload,
    recurrence_survival_from_payload,
    update_dataset_statistics_yml,
    validate_payload_dataset_config,
    write_dataset_statistics_payload,
)


@pytest.mark.parametrize("comparison_state", ["matching", "wrong_inputs", "obsolete"])
def test_legacy_statistics_recover_age_hash_only_from_compatible_comparison(
    tmp_path: Path, comparison_state: str,
) -> None:
    path = tmp_path / "dataset_statistics_values.yml"
    write_dataset_statistics_payload({"metadata": {"input_fingerprint": "inputs"}}, path)
    before = path.read_bytes()
    comparison = {
        "version": COMPARISON_VERSION - 1 if comparison_state == "obsolete" else COMPARISON_VERSION,
        "input_fingerprint": "other" if comparison_state == "wrong_inputs" else "inputs",
        "cohort_hash": "the-cohort-hash",
    }
    (tmp_path / "cohort_comparison_values.json").write_text(json.dumps(comparison))

    if comparison_state == "matching":
        loaded = load_dataset_statistics_payload(path)
        assert loaded["metadata"]["cohort_hash"] == "the-cohort-hash"
    else:
        with pytest.raises(ValueError):
            load_dataset_statistics_payload(path)
    assert path.read_bytes() == before


def test_dataset_statistics_payload_round_trips_updates(tmp_path: Path) -> None:
    summary = DatasetStatisticsSummary(
        selected_test_count=2,
        input_fingerprint="abc123",
        inclusion_box_labels=("Start",),
        inclusion_box_counts=("N = 3",),
        exclusion_box_labels=("Excluded",),
        exclusion_box_counts=("N = 1",),
        split_box_counts=("Train\nN = 2",),
        diagnosis_years=(2020, 2021),
        series_labels=("Included", "Excluded"),
        series_counts=((2, 1), (1, 0)),
        blood_test_names=("Hemoglobin",),
        blood_test_counts=(5,),
        blood_test_locf_counts=(1,),
        blood_test_mice_counts=(2,),
        bloodtest_count_table=pd.DataFrame(
            {
                "display_name": ["Hemoglobin", "Creatinine"],
                "count": [5, 3],
                "locf_count": [1, 0],
                "mice_count": [2, 1],
            }
        ),
        stage_ids=(frozenset({"P1", "P2", "P3"}), frozenset({"P2", "P1"})),
    )
    payload_path = tmp_path / "dataset_statistics_values.yml"

    recurrence_incidence = pd.DataFrame(
        {
            "site": ["Any", "Bone"],
            "years_since_diagnosis": [0.0, 1.0],
            "cumulative_incidence_percent": [0.0, 2.5],
            "confidence_interval_lower_percent": [0.0, 1.0],
            "confidence_interval_upper_percent": [0.0, 4.0],
            "patients_at_risk": [100, 95],
            "target_events": [0, 1],
            "competing_events": [0, 2],
            "censorings": [0, 3],
        }
    )
    recurrence_survival = pd.DataFrame(
        {
            "site": ["Any", "Bone"],
            "years_since_origin": [0.0, 1.0],
            "survival_percent": [100.0, 80.0],
        }
    )
    payload = dataset_statistics_payload(
        summary,
        dataset_config={"min_diagnosis_year": None, "include_recurrence_day_blood_tests": True},
        recurrence_incidence_table=recurrence_incidence,
        recurrence_survival_table=recurrence_survival,
    )
    write_dataset_statistics_payload(payload, payload_path)
    loaded_payload = load_dataset_statistics_payload(payload_path)
    updates = dataset_statistics_updates_from_payload(loaded_payload)

    assert loaded_payload["metadata"]["input_fingerprint"] == "abc123"
    assert loaded_payload["metadata"]["cohort_selection_version"] == COHORT_SELECTION_VERSION
    assert included_patient_ids_from_payload(loaded_payload) == {"P1", "P2"}
    assert updates[("panel_a", "inclusion_box_labels")] == ["Start"]
    assert updates[("panel_a", "exclusion_box_labels")] == ["Excluded"]
    assert loaded_payload["tables"]["bloodtest_count_table"] == [
        {"display_name": "Hemoglobin", "count": 5, "locf_count": 1, "mice_count": 2},
        {"display_name": "Creatinine", "count": 3, "locf_count": 0, "mice_count": 1},
    ]
    pd.testing.assert_frame_equal(
        recurrence_incidence_from_payload(loaded_payload),
        recurrence_incidence,
    )
    pd.testing.assert_frame_equal(
        recurrence_survival_from_payload(loaded_payload),
        recurrence_survival,
    )
    assert updates[("panel_b", "series_counts")] == [[2, 1], [1, 0]]
    assert updates[("panel_c", "blood_test_counts")] == [5]
    assert updates[("panel_c", "blood_test_locf_counts")] == [1]
    assert updates[("panel_c", "blood_test_mice_counts")] == [2]
    validate_payload_dataset_config(payload, {"min_diagnosis_year": None, "include_recurrence_day_blood_tests": True})


def test_dataset_statistics_payload_rejects_stale_dataset_config() -> None:
    summary = DatasetStatisticsSummary(
        selected_test_count=1,
        input_fingerprint="old",
        inclusion_box_labels=("Start",),
        inclusion_box_counts=("N = 1",),
        exclusion_box_labels=("Excluded",),
        exclusion_box_counts=("N = 0",),
        split_box_counts=("Train\nN = 1",),
        diagnosis_years=(2003, 2004),
        series_labels=("Included",),
        series_counts=((1, 1),),
        blood_test_names=("Hemoglobin",),
        blood_test_counts=(2,),
        blood_test_locf_counts=(0,),
        blood_test_mice_counts=(1,),
        bloodtest_count_table=pd.DataFrame(
            {
                "display_name": ["Hemoglobin"],
                "count": [2],
                "locf_count": [0],
                "mice_count": [1],
            }
        ),
        stage_ids=(frozenset({"P1"}),),
    )
    payload = dataset_statistics_payload(
        summary,
        dataset_config={"min_diagnosis_year": None},
    )

    with pytest.raises(ValueError, match="min_diagnosis_year"):
        validate_payload_dataset_config(payload, {"min_diagnosis_year": 2004})


@pytest.mark.parametrize("version", [None, 1])
def test_saved_survival_requires_current_follow_up_rules(version: int | None) -> None:
    payload = {
        "metadata": {"post_recurrence_survival_version": version},
        "panel_e": {"first_recurrence_site_survival": [
            {"site": "Any", "years_since_origin": 0.0, "survival_percent": 100.0},
        ]},
    }

    with pytest.raises(ValueError, match="cohort_statistics"):
        recurrence_survival_from_payload(payload)


@pytest.mark.parametrize("version", [None, 1, 4, 5])
def test_dataset_statistics_rejects_old_exclusion_order(version: int | None) -> None:
    config = {"min_blood_test_types_per_day": 13}
    metadata: dict[str, object] = {"dataset_config": config}
    if version is not None:
        metadata["cohort_selection_version"] = version
    with pytest.raises(ValueError, match="outdated cohort-selection order"):
        validate_payload_dataset_config({"metadata": metadata}, config)

    metadata["cohort_selection_version"] = COHORT_SELECTION_VERSION
    validate_payload_dataset_config({"metadata": metadata}, config)


@pytest.mark.parametrize("version", [None, 1])
def test_dataset_statistics_rejects_old_recurrence_only_inclusion(version: int | None) -> None:
    config = {"include_recurrence_day_blood_tests": True}
    metadata: dict[str, object] = {"dataset_config": config}
    if version is not None:
        metadata["observation_window_version"] = version
    with pytest.raises(ValueError, match="outdated observation-window rules"):
        validate_payload_dataset_config({"metadata": metadata}, config)


def test_update_dataset_statistics_yml_replaces_saved_statistics_lists(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset_statistics.yml"
    config_path.write_text(
        "\n".join(
            [
                "panel_a:",
                '  inclusion_box_labels: ["Start"]',
                "  inclusion_box_counts:",
                '    - "old"',
                "panel_b:",
                "  diagnosis_years:",
                "    - 1999",
                "  series_counts:",
                "    - [1, 2]",
                "panel_c:",
                "  blood_test_counts: []",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    update_dataset_statistics_yml(
        config_path,
        {
            ("panel_a", "inclusion_box_counts"): ["N = 2"],
            ("panel_b", "diagnosis_years"): [2020, 2021],
            ("panel_b", "series_counts"): [[3, 4], [1, 2]],
            ("panel_c", "blood_test_counts"): [10],
        },
    )

    assert config_path.read_text(encoding="utf-8") == "\n".join(
        [
            "panel_a:",
            '  inclusion_box_labels: ["Start"]',
            "  inclusion_box_counts:",
            '    - "N = 2"',
            "panel_b:",
            "  diagnosis_years:",
            "    - 2020",
            "    - 2021",
            "  series_counts:",
            "    - [3, 4]",
            "    - [1, 2]",
            "panel_c:",
            "  blood_test_counts:",
            "    - 10",
            "",
        ]
    )
