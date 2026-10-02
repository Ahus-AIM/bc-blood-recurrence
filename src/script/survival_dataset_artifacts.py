from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from src.data import PatientBloodTestDataset
from src.data.landmark_event_counts import (
    LandmarkEventCountsConfig,
    build_landmark_event_count_frame,
    write_landmark_event_count_outputs,
)
from src.data.population_statistics import (
    PopulationStatisticsConfig,
    write_population_statistics_latex,
)
from src.model.blood_test_imputation_statistics import (
    BloodTestImputationStatistics,
)
from src.script.training_dataset_statistics import write_training_dataset_statistics_png
from src.script.training_cohort_comparison import write_training_cohort_comparison
from src.script.training_recurrence_blood_test_insights import write_training_recurrence_blood_test_insights

ELIGIBLE_PATIENT_FOLDS_FILENAME = "eligible_patient_folds.csv"
MODEL_FOLDS = ("train", "val", "test")


def write_survival_dataset_artifacts(
    dataset: PatientBloodTestDataset,
    output_dir: Path,
    population_config: Mapping[str, object] | None,
    imputation_statistics: BloodTestImputationStatistics | None = None,
    *,
    landmark_event_counts_config: Mapping[str, object] | None,
) -> dict[str, object]:
    """Write eligible folds and dataset reports from one prepared model cohort."""
    return {
        "eligible_patient_folds": write_eligible_patient_folds_csv(
            dataset,
            output_dir,
        ),
        "dataset_statistics": write_training_dataset_statistics_png(
            dataset,
            output_dir,
            imputation_statistics=imputation_statistics,
            population_config=PopulationStatisticsConfig.from_mapping(population_config),
        ),
        "recurrence_blood_test_insights": write_training_recurrence_blood_test_insights(dataset, output_dir),
        "landmark_event_counts": write_training_landmark_event_counts(
            dataset,
            output_dir,
            landmark_event_counts_config,
        ),
        "population_statistics": write_population_statistics_latex(
            dataset,
            PopulationStatisticsConfig.from_mapping(population_config),
            output_dir,
            imputation_statistics=imputation_statistics,
        ),
        "cohort_comparison": write_training_cohort_comparison(
            dataset, output_dir, PopulationStatisticsConfig.from_mapping(population_config),
        ),
    }


def write_training_landmark_event_counts(
    dataset: PatientBloodTestDataset,
    output_dir: Path,
    raw_config: Mapping[str, object] | None,
) -> dict[str, object]:
    """Write the full site/horizon table without rebuilding the prepared dataset."""
    config = LandmarkEventCountsConfig.from_mapping(raw_config, output_dir=output_dir)
    counts = build_landmark_event_count_frame(dataset.config, config, prepared_dataset=dataset)
    return write_landmark_event_count_outputs(counts, config, output_dir)


def write_eligible_patient_folds_csv(
    dataset: PatientBloodTestDataset,
    output_dir: Path,
) -> dict[str, object]:
    """Write one fold assignment for every patient retained by the dataset."""
    rows = tuple(
        (item.patient_id, fold) for fold in MODEL_FOLDS for item in dataset.prepared_data.items_by_split.get(fold, ())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ELIGIBLE_PATIENT_FOLDS_FILENAME
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("ID", "fold"))
        writer.writerows(rows)
    return {
        "path": ELIGIBLE_PATIENT_FOLDS_FILENAME,
        "patient_count": len(rows),
    }
