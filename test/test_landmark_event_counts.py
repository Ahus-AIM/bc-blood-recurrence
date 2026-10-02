from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import src.data.landmark_event_counts as landmark_event_counts
from src.data.dataset import PatientBloodTestDataset
from src.data.dataset_config import PatientDatasetConfig
from src.data.landmark_event_counts import (
    LandmarkEventCountsConfig,
    build_landmark_event_count_frame,
    write_landmark_event_count_outputs,
)
from test.test_patient_bloodtest import blood_row, config, endpoint_row, single_patient_outcome_dataset


def test_landmark_event_counts_include_sites_horizons_and_folds(tmp_path: Path) -> None:
    count_df = build_landmark_event_count_frame(
        event_count_dataset_config(tmp_path),
        event_count_config(tmp_path),
    )

    all_1m = count_row(count_df, "All", "1m")
    assert all_1m["train_event_landmark_count"] == 1
    assert all_1m["train_landmark_count"] == 3
    assert all_1m["train_event_patient_count"] == 1
    assert all_1m["train_patient_count"] == 2
    assert all_1m["train_recurrence_patient_count"] == 1
    assert all_1m["train_event_patient_percent"] == pytest.approx(100.0)
    assert all_1m["val_event_landmark_count"] == 0
    assert all_1m["val_recurrence_patient_count"] == 1
    assert all_1m["val_event_patient_percent"] == pytest.approx(0.0)
    assert all_1m["test_event_landmark_count"] == 1
    assert all_1m["test_recurrence_patient_count"] == 1
    assert all_1m["overall_event_landmark_count"] == 2
    assert all_1m["overall_landmark_count"] == 6
    assert all_1m["overall_event_patient_count"] == 2
    assert all_1m["overall_patient_count"] == 4
    assert all_1m["overall_recurrence_patient_count"] == 3
    assert all_1m["overall_event_patient_percent"] == pytest.approx(200.0 / 3.0)

    all_3m = count_row(count_df, "All", "3m")
    assert all_3m["overall_event_landmark_count"] == 5
    assert all_3m["overall_event_patient_count"] == 3
    assert all_3m["overall_event_patient_percent"] == pytest.approx(100.0)

    liver_3m = count_row(count_df, "Liver", "3m")
    assert liver_3m["train_event_landmark_count"] == 2
    assert liver_3m["val_event_landmark_count"] == 0
    assert liver_3m["test_event_landmark_count"] == 0
    assert liver_3m["train_recurrence_patient_count"] == 1
    assert liver_3m["val_recurrence_patient_count"] == 0
    assert pd.isna(liver_3m["val_event_patient_percent"])

    lung_3m = count_row(count_df, "Lung", "3m")
    assert lung_3m["val_event_landmark_count"] == 1

    other_3m = count_row(count_df, "Other", "3m")
    assert other_3m["test_event_landmark_count"] == 2
    assert other_3m["test_event_patient_count"] == 1


def test_landmark_event_counts_write_csv_and_latex(tmp_path: Path) -> None:
    count_config = event_count_config(tmp_path)
    count_df = build_landmark_event_count_frame(event_count_dataset_config(tmp_path), count_config)

    metadata = write_landmark_event_count_outputs(count_df, count_config, tmp_path)

    assert metadata == {
        "csv_path": "tables/landmark_event_counts.csv",
        "latex_path": "tables/landmark_event_counts.tex",
    }
    written_df = pd.read_csv(tmp_path / "tables" / "landmark_event_counts.csv")
    assert len(written_df) == len(count_df)
    assert "overall_recurrence_patient_count" in written_df.columns
    assert "overall_event_patient_percent" in written_df.columns
    latex = (tmp_path / "tables" / "landmark_event_counts.tex").read_text(encoding="utf-8")
    assert r"\label{tab:landmark\_event\_counts\_test}" in latex
    assert "Site & Dataset fold & 1m & 3m" in latex
    assert (
        r"\multirow[c]{4}{*}{Any} & Train (1) & 1 (100.0\%) & 1 (100.0\%)"
        " \\\\\n"
        " & Validation (1) & 0 (0.0\\%) & 1 (100.0\\%) \\\\\n"
        " & Test (1) & 1 (100.0\\%) & 1 (100.0\\%) \\\\\n"
        " & Overall (3) & 2 (66.7\\%) & 3 (100.0\\%) \\\\\n"
        "\\hline\n"
        r"\multirow[c]{4}{*}{Liver} & Train (1) & 1 (100.0\%) & 1 (100.0\%)"
    ) in latex


def test_landmark_event_counts_prepares_dataset_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def counting_dataset(*args: object, **kwargs: object) -> PatientBloodTestDataset:
        calls.append((args, kwargs))
        return PatientBloodTestDataset(*args, **kwargs)

    monkeypatch.setattr(landmark_event_counts, "PatientBloodTestDataset", counting_dataset)

    build_landmark_event_count_frame(
        event_count_dataset_config(tmp_path),
        event_count_config(tmp_path),
    )

    assert len(calls) == 1


def test_landmark_event_counts_rejects_mismatched_site_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="site_labels.*site_regexes"):
        LandmarkEventCountsConfig.from_mapping(
            {
                "site_labels": ["Liver"],
                "site_regexes": [r"C787\d*", r"C780\d*"],
                "horizons": [{"label": "1m", "days": 30}],
            },
            output_dir=tmp_path,
        )


def test_landmark_event_counts_reuses_prepared_patients_and_landmark_days(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_config = event_count_dataset_config(tmp_path)
    original = PatientBloodTestDataset(dataset_config)
    items = original.prepared_data.items_by_split
    prepared = PatientBloodTestDataset(
        dataset_config,
        prepared_data=replace(
            original.prepared_data,
            items_by_split={
                **items,
                "train": tuple(replace(item, valid_landmark_days=(40,)) for item in items["train"]),
                "test": (),
            },
        ),
    )

    def unexpected_dataset(*_args: object, **_kwargs: object) -> PatientBloodTestDataset:
        pytest.fail("Event-count reports must not rebuild the prepared dataset.")

    monkeypatch.setattr(landmark_event_counts, "PatientBloodTestDataset", unexpected_dataset)
    counts = build_landmark_event_count_frame(
        dataset_config,
        event_count_config(tmp_path),
        prepared_dataset=prepared,
    )

    all_1m = count_row(counts, "All", "1m")
    assert all_1m["train_landmark_count"] == 2
    assert all_1m["train_event_patient_count"] == 0
    assert all_1m["test_patient_count"] == 0
    assert count_row(counts, "Liver", "3m")["train_event_patient_count"] == 1


def test_landmark_counts_apply_other_cancer_tie_rule(tmp_path: Path) -> None:
    dataset = single_patient_outcome_dataset(
        tmp_path,
        recurrence_day=100,
        other_cancer_day=100,
        other_cancer_enabled=True,
    )
    count_df = build_landmark_event_count_frame(
        dataset.config,
        event_count_config(tmp_path),
    )
    all_3m = count_row(count_df, "All", "3m")

    assert all_3m["overall_recurrence_patient_count"] == 0
    assert all_3m["overall_event_patient_count"] == 0


def event_count_dataset_config(tmp_path: Path) -> PatientDatasetConfig:
    raw_config = {
        **config(
            tmp_path,
            [
                blood_row("train_liver", 40, test_name="A", unit="u"),
                blood_row("train_liver", 80, test_name="A", unit="u"),
                blood_row("train_censored", 40, test_name="A", unit="u"),
                blood_row("validation_lung", 40, test_name="A", unit="u"),
                blood_row("test_other", 40, test_name="A", unit="u"),
                blood_row("test_other", 80, test_name="A", unit="u"),
            ],
            [
                endpoint_row(
                    "train_liver",
                    diagnosis_to_ddfs=100,
                    ddfs_code="C787",
                    last_entry_days=400,
                ),
                endpoint_row("train_censored", last_entry_days=400),
                endpoint_row(
                    "validation_lung",
                    diagnosis_to_ddfs=100,
                    ddfs_code="C780",
                    last_entry_days=400,
                ),
                endpoint_row(
                    "test_other",
                    diagnosis_to_ddfs=100,
                    ddfs_code="C790",
                    last_entry_days=400,
                ),
            ],
            split_by_patient={
                "train_liver": "train",
                "train_censored": "train",
                "validation_lung": "val",
                "test_other": "test",
            },
        ),
        "selected_blood_tests": ["A|||u"],
        "min_landmark_day": 30,
        "build_patient_timeline_tensors": False,
    }
    return PatientDatasetConfig.from_mapping(raw_config)


def event_count_config(tmp_path: Path) -> LandmarkEventCountsConfig:
    return LandmarkEventCountsConfig.from_mapping(
        {
            "output_csv_path": "tables/landmark_event_counts.csv",
            "output_latex_path": "tables/landmark_event_counts.tex",
            "table_label": "tab:landmark_event_counts_test",
            "include_all_site": True,
            "all_site_label": "All",
            "other_label": "Other",
            "site_labels": ["Liver", "Lung"],
            "site_regexes": [r"C787\d*", r"C780\d*"],
            "horizons": [
                {"label": "1m", "days": 30},
                {"label": "3m", "days": 90},
            ],
        },
        output_dir=tmp_path,
    )


def count_row(count_df: pd.DataFrame, site_label: str, horizon_label: str) -> pd.Series:
    rows = count_df[
        (count_df["site_label"] == site_label)
        & (count_df["horizon_label"] == horizon_label)
    ]
    assert len(rows) == 1
    return rows.iloc[0]
