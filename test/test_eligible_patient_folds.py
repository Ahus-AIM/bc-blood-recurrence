from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from src.script.survival_dataset_artifacts import write_eligible_patient_folds_csv


def test_eligible_patient_folds_csv_contains_only_retained_patients(
    tmp_path: Path,
) -> None:
    retained = {
        "train": (SimpleNamespace(patient_id="train_patient"),),
        "val": (SimpleNamespace(patient_id="validation_patient"),),
        "test": (SimpleNamespace(patient_id="test_patient"),),
    }
    dataset = SimpleNamespace(
        prepared_data=SimpleNamespace(
            items_by_split=retained,
            discarded_patient_items_by_id={
                "filtered_patient": SimpleNamespace(patient_id="filtered_patient")
            },
        )
    )

    metadata = write_eligible_patient_folds_csv(dataset, tmp_path)  # type: ignore[arg-type]

    with (tmp_path / "eligible_patient_folds.csv").open(
        encoding="utf-8",
        newline="",
    ) as input_file:
        rows = tuple(csv.DictReader(input_file))
    assert rows == (
        {"ID": "train_patient", "fold": "train"},
        {"ID": "validation_patient", "fold": "val"},
        {"ID": "test_patient", "fold": "test"},
    )
    assert metadata == {
        "path": "eligible_patient_folds.csv",
        "patient_count": 3,
    }
