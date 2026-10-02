from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import pandas as pd

from src.data.blood_test_display import (
    display_title_for_test_key,
    load_test_name_translation as load_shared_test_name_translation,
)
from src.data.cohort_selection import (
    STAGE_BC_DIAGNOSIS_WINDOW,
    STAGE_DIAGNOSIS_YEAR,
    STAGE_EARLY_OTHER_CANCER,
    STAGE_EARLY_RECURRENCE,
    STAGE_EARLY_MODEL_MEASUREMENTS,
    STAGE_INITIAL,
    STAGE_MIN_BLOOD_TEST_TYPES,
    STAGE_SPARSE_MODEL_MEASUREMENTS,
    STAGE_MIN_LANDMARK_DAY,
    STAGE_OBSERVATION_WINDOW,
    STAGE_PRE_DIAGNOSIS_OTHER_CANCER,
    STAGE_PRE_DIAGNOSIS_RECURRENCE,
    STAGE_PRIOR_BC_SURGERY,
)
from src.data.dataset import PatientBloodTestDataset
from src.data.dataset_config import PatientDatasetConfig
from src.data.measurement_selection import selected_cohort_measurements
from src.data.patient_parsing import load_endpoint_patients
from src.data.recurrence_site_incidence import load_item_endpoint_site_rows
from src.data.types import (
    PatientBloodTestItem,
    PatientCohortStage,
    PatientRecord,
    PreparedPatientData,
)

if TYPE_CHECKING:
    from src.model.blood_test_imputation_statistics import (
        BloodTestImputationStatistics,
    )

DATASET_STATISTICS_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetStatisticsSummary:
    """Notebook-ready figure values derived from the shared model cohort flow.

    Panel A fields describe sequential patient and deduplicated blood-test-row
    counts. Panel B fields contain diagnosis-year stacks from 2004 onward.
    Panel C fields combine retained measurements with LOCF/MICE current values
    and MICE-imputed past inputs used for changes.
    `stage_ids` retains the patient membership used at every Panel A stage.
    `flow_footnotes` explains measurement exclusions using the active dataset settings.
    """

    selected_test_count: int
    input_fingerprint: str
    inclusion_box_labels: tuple[str, ...]
    inclusion_box_counts: tuple[str, ...]
    exclusion_box_labels: tuple[str, ...]
    exclusion_box_counts: tuple[str, ...]
    split_box_counts: tuple[str, ...]
    diagnosis_years: tuple[int, ...]
    series_labels: tuple[str, ...]
    series_counts: tuple[tuple[int, ...], ...]
    blood_test_names: tuple[str, ...]
    blood_test_counts: tuple[int, ...]
    blood_test_locf_counts: tuple[int, ...]
    blood_test_mice_counts: tuple[int, ...]
    bloodtest_count_table: pd.DataFrame
    stage_ids: tuple[frozenset[str], ...]
    flow_footnotes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetStatisticsContext:
    """Prepared shared dataset and endpoint patients used for figure summaries."""

    dataset: PatientBloodTestDataset
    endpoint_patients: dict[str, PatientRecord]


def build_dataset_statistics_summary(
    config: PatientDatasetConfig,
    *,
    prepared_dataset: PatientBloodTestDataset | None = None,
    test_name_translation: Mapping[str, str] | None = None,
    max_bars: int | None = None,
    imputation_statistics: BloodTestImputationStatistics | None = None,
) -> DatasetStatisticsSummary:
    context = dataset_statistics_context(config, prepared_dataset)
    prepared_data = context.dataset.prepared_data
    stages = tuple(
        stage
        for stage in prepared_data.cohort_stages
        if stage.key not in {STAGE_MIN_BLOOD_TEST_TYPES, STAGE_SPARSE_MODEL_MEASUREMENTS}
        or config.min_blood_test_types_per_day != 1
    )
    stage_ids = tuple(stage.patient_ids for stage in stages)
    stage_exclusions = exclusion_stage_ids(stage_ids)
    labels = tuple(inclusion_label(stage.key, config) for stage in stages)
    exclusion_labels = tuple(
        exclusion_label(stage.key, config) for stage in stages[1:]
    )
    valid_measurements = selected_cohort_measurements(prepared_data)
    translation = {} if test_name_translation is None else dict(test_name_translation)
    bloodtest_count_table = bloodtest_count_frame(
        valid_measurements,
        translation,
        imputation_statistics,
    )
    bloodtest_plot_table = (
        bloodtest_count_table
        if max_bars is None
        else bloodtest_count_table.head(max_bars)
    )
    series_labels, diagnosis_years, series_counts = diagnosis_year_stack(
        stages,
        stage_exclusions,
        context.endpoint_patients,
        config,
    )
    return DatasetStatisticsSummary(
        selected_test_count=len(prepared_data.selected_test_keys),
        input_fingerprint=prepared_data.input_fingerprint,
        inclusion_box_labels=labels,
        inclusion_box_counts=tuple(stage_count_text(stage) for stage in stages),
        exclusion_box_labels=exclusion_labels,
        exclusion_box_counts=tuple(
            count_with_bloodtests_text(
                len(excluded_ids),
                stages[index].bloodtest_count - stages[index + 1].bloodtest_count,
            )
            for index, excluded_ids in enumerate(stage_exclusions)
        ),
        split_box_counts=split_box_counts(
            eligible_items_by_split(prepared_data),
            valid_measurements,
        ),
        diagnosis_years=diagnosis_years,
        series_labels=series_labels,
        series_counts=series_counts,
        blood_test_names=tuple(bloodtest_plot_table["display_name"].astype(str)),
        blood_test_counts=tuple(int(count) for count in bloodtest_plot_table["count"]),
        blood_test_locf_counts=tuple(
            int(count) for count in bloodtest_plot_table["locf_count"]
        ),
        blood_test_mice_counts=tuple(
            int(count) for count in bloodtest_plot_table["mice_count"]
        ),
        bloodtest_count_table=bloodtest_count_table,
        stage_ids=stage_ids,
        flow_footnotes=(observation_window_note(config),),
    )


def load_test_name_translation(translation_csv_path: Path | None) -> dict[str, str]:
    return load_shared_test_name_translation(translation_csv_path)


def dataset_statistics_context(
    config: PatientDatasetConfig,
    prepared_dataset: PatientBloodTestDataset | None = None,
) -> DatasetStatisticsContext:
    return DatasetStatisticsContext(
        dataset=prepared_dataset or PatientBloodTestDataset(config),
        endpoint_patients=load_endpoint_patients(config.endpoint_csv_path),
    )


def count_text(count: int) -> str:
    return f"N = {count:,}"


def count_with_bloodtests_text(patient_count: int, bloodtest_count: int) -> str:
    return f"{count_text(patient_count)}\nblood tests = {bloodtest_count:,}"


def stage_count_text(stage: PatientCohortStage) -> str:
    return count_with_bloodtests_text(len(stage.patient_ids), stage.bloodtest_count)


def eligible_items_by_split(
    prepared_data: PreparedPatientData,
) -> dict[str, tuple[PatientBloodTestItem, ...]]:
    return {
        split: tuple(prepared_data.items_by_split.get(split, ()))
        for split in DATASET_STATISTICS_SPLITS
    }


def all_eligible_items(prepared_data: PreparedPatientData) -> tuple[PatientBloodTestItem, ...]:
    return tuple(
        item
        for split in DATASET_STATISTICS_SPLITS
        for item in prepared_data.items_by_split.get(split, ())
    )


def prepared_recurrence_site_rows(dataset: PatientBloodTestDataset) -> pd.DataFrame:
    """Return patient-level first-recurrence rows for dataset-statistics panels."""
    return load_item_endpoint_site_rows(
        all_eligible_items(dataset.prepared_data),
        dataset.config,
    )


def selected_model_measurements(prepared_data: PreparedPatientData) -> pd.DataFrame:
    records = [
        (item.patient_id, day)
        for item in all_eligible_items(prepared_data)
        for day in item.valid_landmark_days
    ]
    if not records:
        return prepared_data.measurement_df.head(0)
    valid_days = pd.DataFrame.from_records(
        records,
        columns=["patient_id", "days_since_diagnosis"],
    ).drop_duplicates()
    return prepared_data.measurement_df.merge(
        valid_days,
        on=["patient_id", "days_since_diagnosis"],
        how="inner",
    )


def exclusion_stage_ids(
    stage_ids: tuple[frozenset[str], ...],
) -> tuple[frozenset[str], ...]:
    return tuple(
        stage_ids[index] - stage_ids[index + 1]
        for index in range(len(stage_ids) - 1)
    )


def inclusion_label(stage_key: str, config: PatientDatasetConfig) -> str:
    labels = {
        STAGE_INITIAL: (
            "Females diagnosed with BC at Ahus\n\n1.12.1998 - 08.06.2026"
        ),
        STAGE_DIAGNOSIS_YEAR: f"Initial BC diagnosis ≥ {config.min_diagnosis_year}",
        STAGE_PRIOR_BC_SURGERY: "No prior BC surgery",
        STAGE_PRE_DIAGNOSIS_OTHER_CANCER: "No other cancer before BC diagnosis",
        STAGE_PRE_DIAGNOSIS_RECURRENCE: "No distant recurrence before BC diagnosis",
        STAGE_EARLY_OTHER_CANCER: early_other_cancer_stage_label(
            config,
            excluded=False,
        ),
        STAGE_EARLY_RECURRENCE: (
            f"No distant recurrence ≤ {config.min_landmark_day - 1} days after BC diagnosis"
        ),
        STAGE_BC_DIAGNOSIS_WINDOW: bc_diagnosis_window_label(config),
        STAGE_MIN_BLOOD_TEST_TYPES: "Patients with at least one qualifying day:\n" + minimum_blood_test_types_day_label(
            config.min_blood_test_types_per_day, config.min_landmark_day,
        ).replace("Days", "days", 1),
        STAGE_SPARSE_MODEL_MEASUREMENTS: minimum_blood_test_types_day_label(
            config.min_blood_test_types_per_day,
            config.min_landmark_day if config.include_blood_test_change else None,
        ),
        STAGE_MIN_LANDMARK_DAY: minimum_landmark_inclusion_label(
            config.min_landmark_day,
        ),
        STAGE_OBSERVATION_WINDOW: observation_window_label(config, excluded=False),
        STAGE_EARLY_MODEL_MEASUREMENTS: f"Blood tests {landmark_day_label(config.min_landmark_day)}",
    }
    return labels[stage_key]


def exclusion_label(stage_key: str, config: PatientDatasetConfig) -> str:
    labels = {
        STAGE_DIAGNOSIS_YEAR: f"Initial BC diagnosis before {config.min_diagnosis_year}",
        STAGE_PRIOR_BC_SURGERY: "Prior BC surgery",
        STAGE_PRE_DIAGNOSIS_OTHER_CANCER: "Other cancer before BC diagnosis",
        STAGE_PRE_DIAGNOSIS_RECURRENCE: "Distant recurrence before BC diagnosis",
        STAGE_EARLY_OTHER_CANCER: early_other_cancer_stage_label(
            config,
            excluded=True,
        ),
        STAGE_EARLY_RECURRENCE: (
            f"Distant recurrence ≤ {config.min_landmark_day - 1} days after BC diagnosis"
        ),
        STAGE_BC_DIAGNOSIS_WINDOW: "Blood test unrelated to BC checkup",
        STAGE_MIN_BLOOD_TEST_TYPES: minimum_blood_test_types_exclusion_label(
            config.min_blood_test_types_per_day, config.min_landmark_day,
        ),
        STAGE_SPARSE_MODEL_MEASUREMENTS: minimum_blood_test_types_day_label(
            config.min_blood_test_types_per_day,
            config.min_landmark_day if config.include_blood_test_change else None,
            excluded=True,
        ),
        STAGE_MIN_LANDMARK_DAY: minimum_landmark_exclusion_label(
            config.min_landmark_day
        ),
        STAGE_OBSERVATION_WINDOW: observation_window_label(config, excluded=True),
        STAGE_EARLY_MODEL_MEASUREMENTS: (
            f"Blood tests ≤ {config.min_landmark_day - 1} days after BC diagnosis"
            if config.min_landmark_day else "Blood tests before BC diagnosis"
        ),
    }
    return labels[stage_key]


def early_other_cancer_stage_label(
    config: PatientDatasetConfig,
    *,
    excluded: bool,
) -> str:
    prefix = "Other cancer" if excluded else "No other cancer"
    return f"{prefix} ≤ {config.min_landmark_day - 1} days after BC diagnosis"


def observation_window_label(
    config: PatientDatasetConfig,
    *,
    excluded: bool,
) -> str:
    if excluded:
        timing = "after" if config.include_recurrence_day_blood_tests else "on or after"
    else:
        timing = "on or before" if config.include_recurrence_day_blood_tests else "before"
    marker = "¹" if excluded else ""
    return f"Blood tests {timing} (competing) event or censoring{marker}"


def observation_window_note(config: PatientDatasetConfig) -> str:
    """Explain the observation cutoff and subsequent patient-eligibility check."""
    timing = "after" if config.include_recurrence_day_blood_tests else "on or after"
    outcomes = "first distant recurrence, death or last observation"
    if config.exclude_and_censor_other_cancer_patients:
        outcomes = "another cancer, " + outcomes
    return (
        f"¹ Blood test results {timing} {outcomes}, whichever occurred first, were removed. "
        "Patients with no remaining blood test results were also excluded."
    )


def bc_diagnosis_window_label(config: PatientDatasetConfig) -> str:
    if (
        config.bc_diagnosis_window_days_before == 1
        and config.bc_diagnosis_window_days_after == 0
    ):
        return "Blood test on-/one day before a BC checkup"
    return "Blood tests within the configured C50 diagnosis window"


def minimum_blood_test_types_day_label(
    minimum_count: int | None, min_landmark_day: int | None, *, excluded: bool = False,
) -> str:
    """Omit the day restriction only when earlier model rows have already been removed."""
    prefix = "Days " if min_landmark_day is None else f"Days {landmark_day_label(min_landmark_day)}\n"
    count = "the configured number of" if minimum_count is None else f"> {minimum_count - 1}"
    if excluded and minimum_count is not None:
        count = f"≤ {minimum_count - 1}"
    return f"{prefix}with {count} different blood tests"


def minimum_blood_test_types_exclusion_label(minimum_count: int | None, min_landmark_day: int) -> str:
    return "No " + minimum_blood_test_types_day_label(minimum_count, min_landmark_day).replace("Days", "days", 1)


def landmark_day_label(min_landmark_day: int) -> str:
    if min_landmark_day == 0:
        return "on or after BC diagnosis"
    return f"> {min_landmark_day - 1} days after BC diagnosis"


def minimum_landmark_inclusion_label(min_landmark_day: int) -> str:
    return f"Any blood tests {landmark_day_label(min_landmark_day)}"


def minimum_landmark_exclusion_label(min_landmark_day: int) -> str:
    return f"No blood tests {landmark_day_label(min_landmark_day)}"


def diagnosis_year_stack(
    stages: tuple[PatientCohortStage, ...],
    stage_exclusions: tuple[frozenset[str], ...],
    endpoint_patients: Mapping[str, PatientRecord],
    config: PatientDatasetConfig,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    excluded_by_label = {
        exclusion_label(stage.key, config): ids
        for stage, ids in zip(stages[1:], stage_exclusions, strict=True)
        if stage.key not in {STAGE_DIAGNOSIS_YEAR, STAGE_EARLY_MODEL_MEASUREMENTS, STAGE_SPARSE_MODEL_MEASUREMENTS}
    }
    final_ids = stages[-1].patient_ids
    category_ids = {"Included": final_ids, **excluded_by_label}
    years = diagnosis_years(endpoint_patients, config.min_diagnosis_year)
    labels = tuple(category_ids)
    counts = tuple(
        counts_by_year(category_ids[label], years, endpoint_patients)
        for label in labels
    )
    return labels, years, counts


def diagnosis_years(
    endpoint_patients: Mapping[str, PatientRecord],
    min_diagnosis_year: int | None,
) -> tuple[int, ...]:
    observed_years = tuple(
        patient.diagnosis_year
        for patient in endpoint_patients.values()
        if min_diagnosis_year is None or patient.diagnosis_year >= min_diagnosis_year
    )
    if not observed_years:
        return ()
    first_year = min(observed_years) if min_diagnosis_year is None else min_diagnosis_year
    return tuple(range(first_year, max(observed_years) + 1))


def counts_by_year(
    patient_ids: frozenset[str],
    years: tuple[int, ...],
    endpoint_patients: Mapping[str, PatientRecord],
) -> tuple[int, ...]:
    counts = Counter(
        endpoint_patients[patient_id].diagnosis_year
        for patient_id in patient_ids
        if patient_id in endpoint_patients
    )
    return tuple(int(counts.get(year, 0)) for year in years)


def split_box_counts(
    items_by_split: Mapping[str, tuple[PatientBloodTestItem, ...]],
    valid_measurements: pd.DataFrame,
) -> tuple[str, ...]:
    bloodtest_counts = split_bloodtest_counts(items_by_split, valid_measurements)
    counts = []
    for split in DATASET_STATISTICS_SPLITS:
        items = items_by_split.get(split, ())
        years = [item.diagnosis_year for item in items]
        count = count_with_bloodtests_text(len(items), bloodtest_counts[split])
        counts.append(f"{min(years)} - {max(years)}\n{count}" if years else count)
    return tuple(counts)


def split_bloodtest_counts(
    items_by_split: Mapping[str, tuple[PatientBloodTestItem, ...]],
    valid_measurements: pd.DataFrame,
) -> dict[str, int]:
    split_by_patient = {
        item.patient_id: split
        for split, items in items_by_split.items()
        for item in items
    }
    if valid_measurements.empty or not split_by_patient:
        return {split: 0 for split in DATASET_STATISTICS_SPLITS}
    split_counts = valid_measurements["patient_id"].map(split_by_patient).value_counts()
    return {
        split: int(split_counts.get(split, 0))
        for split in DATASET_STATISTICS_SPLITS
    }


def bloodtest_count_frame(
    valid_measurements: pd.DataFrame,
    test_name_translation: Mapping[str, str],
    imputation_statistics: BloodTestImputationStatistics | None = None,
) -> pd.DataFrame:
    if imputation_statistics is not None:
        return imputation_count_frame(
            imputation_statistics,
            test_name_translation,
        )
    if valid_measurements.empty:
        return pd.DataFrame(
            columns=["display_name", "count", "locf_count", "mice_count"]
        )
    count_df = valid_measurements.copy()
    count_df["display_name"] = count_df["test_key"].map(
        lambda test_key: display_title_for_test_key(
            str(test_key),
            test_name_translation,
        )
    )
    counts = (
        count_df.groupby("display_name")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "display_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    counts["locf_count"] = 0
    counts["mice_count"] = 0
    return counts


def imputation_count_frame(
    statistics: BloodTestImputationStatistics,
    test_name_translation: Mapping[str, str],
) -> pd.DataFrame:
    records = [
        {
            "display_name": display_title_for_test_key(
                test_key,
                test_name_translation,
            ),
            "count": counts.measured,
            "locf_count": counts.locf,
            "mice_count": counts.mice,
        }
        for test_key, counts in statistics.by_test_key.items()
    ]
    if not records:
        return pd.DataFrame(
            columns=["display_name", "count", "locf_count", "mice_count"]
        )
    return (
        pd.DataFrame.from_records(records)
        .groupby("display_name", as_index=False)
        .sum()
        .sort_values(["count", "display_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
