from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, cast

from src.config import resolve_path
from src.data.prior_bc_exclusions import DEFAULT_PRIOR_BC_SURGERY_CODE_REGEX
from src.data.temporal_splits import validate_temporal_cutoffs
from src.time_units import DAYS_PER_WEEK, DAYS_PER_YEAR

PersonSampleWeighting = Literal["person_uniform", "person_time_weighted"]
SampleWeighting = PersonSampleWeighting | Literal["landmark_uniform"]
SAMPLE_WEIGHTING_MODES = ("person_uniform", "person_time_weighted", "landmark_uniform")


@dataclass(frozen=True)
class PatientDatasetConfig:
    """Dataset sources, endpoint, sampling, weighting, feature, and timeline settings.

    ``training_sample_weighting`` selects training landmark weights.
    ``bootstrap_sample_weighting`` selects landmark bootstrap sampling.
    Both accept person-uniform, person-time-weighted, or landmark-uniform
    modes and default to ``person_uniform``.
    Patient-based test bootstraps draw as many times as there are patients with
    eligible landmarks in that pool. ``test_bootstrap_samples`` applies only to
    landmark-uniform sampling and is retained when reading older saved configs.
    Training and validation bootstrap sizes remain configurable.
    ``include_recurrence_day_blood_tests`` retains its original config name but
    includes every observation end day: first input recurrence, death, other
    cancer, or last observation. It also includes that day in time weighting.
    ``include_blood_test_change`` adds completed current minus completed past values;
    ``blood_test_change_min_days`` and ``blood_test_change_max_days`` bound
    the prior measurement's age inclusively.
    Change features retain earlier observed history, including pre-diagnosis
    tests, without making those days landmarks. Recurrence reports always use
    eligible measurements before model-only early-row and sparse-day exclusions.
    ``train_end_year`` and ``test_start_year`` bound the temporal splits; final
    eligible patients must match the source assignment for their diagnosis year.
    """

    split: str
    bloodtest_csv_path: Path
    endpoint_csv_path: Path
    split_csv_path: Path
    bc_diagnosis_patient_age_csv_path: Path | None
    diagnosis_procedure_csv_path: Path | None
    bc_diagnosis_days_csv_path: Path | None
    exclude_prior_bc_surgery_patients: bool
    prior_bc_surgery_code_regex: str
    exclude_and_censor_other_cancer_patients: bool
    other_cancer_code_filter_regex: str
    restrict_bloodtests_to_bc_diagnosis_window: bool
    bc_diagnosis_window_days_before: int
    bc_diagnosis_window_days_after: int
    input_recurrence_code_filter_regex: str
    endpoint_ddfs_code_filter_regex: str
    recurrence_endpoint_name: str
    num_most_common_blood_tests: int
    selected_blood_tests: tuple[str, ...]
    censoring_horizon_days: int
    min_landmark_day: int
    include_recurrence_day_blood_tests: bool
    min_blood_test_types_per_day: int | None
    min_diagnosis_year: int | None
    train_end_year: int
    test_start_year: int
    training_sample_weighting: SampleWeighting
    bootstrap_sample_weighting: SampleWeighting
    random_seed: int
    train_bootstrap_runs: int
    train_bootstrap_samples: int
    validation_bootstrap_runs: int
    validation_bootstrap_samples: int
    test_bootstrap_runs: int
    test_bootstrap_samples: int
    include_diagnosis_age: bool
    require_diagnosis_age: bool
    include_time_since_diagnosis: bool
    include_blood_test_change: bool
    blood_test_change_min_days: int
    blood_test_change_max_days: int
    include_recent_diagnoses: bool
    num_most_common_recent_diagnoses: int
    recent_diagnosis_lookback_days: int
    include_landmark_day_recent_diagnoses: bool
    build_patient_timeline_tensors: bool
    set_all_bloodtest_values_to_one: bool

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object],
        *,
        config_dir: Path | None = None,
    ) -> PatientDatasetConfig:
        validate_removed_rate_of_change(config)
        if "include_blood_tests_before_min_landmark_day" in config:
            raise ValueError(
                "`dataset.include_blood_tests_before_min_landmark_day` was removed. "
                "Remove it for new runs: recurrence reports always include earlier history, "
                "and model history is controlled by `dataset.include_blood_test_change`. "
                "Historical run configurations must not be silently migrated."
            )
        if "max_landmark_missing_fraction" in config:
            raise ValueError(
                "`dataset.max_landmark_missing_fraction` was removed; use "
                "`dataset.min_blood_test_types_per_day` instead."
            )
        base_dir = config_dir or Path.cwd()
        diagnosis_procedure_csv_path = optional_path(config.get("diagnosis_procedure_csv_path"), base_dir)
        bc_diagnosis_days_csv_path = optional_path(config.get("bc_diagnosis_days_csv_path"), base_dir)
        exclude_prior_bc_surgery_patients = bool(
            config.get("exclude_prior_bc_surgery_patients", False)
        )
        prior_bc_surgery_code_regex = str(
            config.get("prior_bc_surgery_code_regex", DEFAULT_PRIOR_BC_SURGERY_CODE_REGEX)
        )
        exclude_and_censor_other_cancer_patients = bool(
            config.get("exclude_and_censor_other_cancer_patients", False)
        )
        other_cancer_code_filter_regex = str(config.get("other_cancer_code_filter_regex", ""))
        restrict_bloodtests_to_bc_diagnosis_window = bool(
            config.get("restrict_bloodtests_to_bc_diagnosis_window", False)
        )
        bc_diagnosis_window_days_before = int(config.get("bc_diagnosis_window_days_before", 7))
        bc_diagnosis_window_days_after = int(config.get("bc_diagnosis_window_days_after", 7))
        include_recent_diagnoses = bool(config.get("include_recent_diagnoses", False))
        num_most_common_recent_diagnoses = int(config.get("num_most_common_recent_diagnoses", 20))
        recent_diagnosis_lookback_days = int(config.get("recent_diagnosis_lookback_days", DAYS_PER_YEAR))
        min_landmark_day = int(config.get("min_landmark_day", 0))
        min_blood_test_types_per_day = optional_int(
            config.get("min_blood_test_types_per_day")
        )
        min_diagnosis_year = optional_int(config.get("min_diagnosis_year"))
        train_end_year = int(config.get("train_end_year", 2016))
        test_start_year = int(config.get("test_start_year", 2022))
        validate_temporal_cutoffs(train_end_year, test_start_year)
        change_min_days = int(config.get("blood_test_change_min_days", DAYS_PER_WEEK))
        change_max_days = int(config.get("blood_test_change_max_days", DAYS_PER_YEAR))
        validate_blood_test_change_window(change_min_days, change_max_days)
        validate_bc_diagnosis_window_config(
            restrict_bloodtests_to_bc_diagnosis_window,
            bc_diagnosis_days_csv_path,
            bc_diagnosis_window_days_before,
            bc_diagnosis_window_days_after,
        )
        validate_recent_diagnosis_config(
            include_recent_diagnoses,
            diagnosis_procedure_csv_path,
            num_most_common_recent_diagnoses,
            recent_diagnosis_lookback_days,
        )
        validate_prior_bc_surgery_exclusion_config(
            exclude_prior_bc_surgery_patients,
            diagnosis_procedure_csv_path,
            prior_bc_surgery_code_regex,
        )
        validate_other_cancer_config(
            exclude_and_censor_other_cancer_patients,
            diagnosis_procedure_csv_path,
            other_cancer_code_filter_regex,
        )
        validate_landmark_config(min_landmark_day)
        validate_min_blood_test_types_per_day(min_blood_test_types_per_day)
        validate_min_diagnosis_year(min_diagnosis_year)
        input_recurrence_regex = str(
            config.get("input_recurrence_code_filter_regex", r"(?:C78\d*)|(?:C79\d*)")
        )
        endpoint_regex = str(
            config.get("endpoint_ddfs_code_filter_regex", r"(?:C78\d*)|(?:C79\d*)")
        )
        re.compile(input_recurrence_regex)
        re.compile(endpoint_regex)
        return cls(
            split=normalize_split(str(config.get("split", "train"))),
            bloodtest_csv_path=required_path(config, "bloodtest_csv_path", base_dir),
            endpoint_csv_path=required_path(config, "endpoint_csv_path", base_dir),
            split_csv_path=required_path(config, "split_csv_path", base_dir),
            bc_diagnosis_patient_age_csv_path=optional_path(
                config.get("bc_diagnosis_patient_age_csv_path"), base_dir
            ),
            diagnosis_procedure_csv_path=diagnosis_procedure_csv_path,
            bc_diagnosis_days_csv_path=bc_diagnosis_days_csv_path,
            exclude_prior_bc_surgery_patients=exclude_prior_bc_surgery_patients,
            prior_bc_surgery_code_regex=prior_bc_surgery_code_regex,
            exclude_and_censor_other_cancer_patients=exclude_and_censor_other_cancer_patients,
            other_cancer_code_filter_regex=other_cancer_code_filter_regex,
            restrict_bloodtests_to_bc_diagnosis_window=restrict_bloodtests_to_bc_diagnosis_window,
            bc_diagnosis_window_days_before=bc_diagnosis_window_days_before,
            bc_diagnosis_window_days_after=bc_diagnosis_window_days_after,
            input_recurrence_code_filter_regex=input_recurrence_regex,
            endpoint_ddfs_code_filter_regex=endpoint_regex,
            recurrence_endpoint_name=str(config.get("recurrence_endpoint_name", "distant_recurrence")),
            num_most_common_blood_tests=int(config.get("num_most_common_blood_tests", 20)),
            selected_blood_tests=parse_selected_blood_tests(config.get("selected_blood_tests", ())),
            censoring_horizon_days=int(config.get("censoring_horizon_days", DAYS_PER_YEAR * 5)),
            min_landmark_day=min_landmark_day,
            include_recurrence_day_blood_tests=bool(config.get("include_recurrence_day_blood_tests", False)),
            min_blood_test_types_per_day=min_blood_test_types_per_day,
            min_diagnosis_year=min_diagnosis_year,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
            training_sample_weighting=parse_training_sample_weighting(
                config.get("training_sample_weighting", "person_uniform")
            ),
            bootstrap_sample_weighting=parse_bootstrap_sample_weighting(
                config.get("bootstrap_sample_weighting", "person_uniform")
            ),
            random_seed=int(config.get("random_seed", 42)),
            train_bootstrap_runs=int(config.get("train_bootstrap_runs", 1)),
            train_bootstrap_samples=int(config.get("train_bootstrap_samples", 1000)),
            validation_bootstrap_runs=int(config.get("validation_bootstrap_runs", 1)),
            validation_bootstrap_samples=int(config.get("validation_bootstrap_samples", 1000)),
            test_bootstrap_runs=int(config.get("test_bootstrap_runs", 1)),
            test_bootstrap_samples=int(config.get("test_bootstrap_samples", 1000)),
            include_diagnosis_age=bool(config.get("include_diagnosis_age", False)),
            require_diagnosis_age=bool(config.get("require_diagnosis_age", False)),
            include_time_since_diagnosis=bool(config.get("include_time_since_diagnosis", False)),
            include_blood_test_change=bool(config.get("include_blood_test_change", False)),
            blood_test_change_min_days=change_min_days,
            blood_test_change_max_days=change_max_days,
            include_recent_diagnoses=include_recent_diagnoses,
            num_most_common_recent_diagnoses=num_most_common_recent_diagnoses,
            recent_diagnosis_lookback_days=recent_diagnosis_lookback_days,
            include_landmark_day_recent_diagnoses=bool(
                config.get("include_landmark_day_recent_diagnoses", False)
            ),
            build_patient_timeline_tensors=bool(config.get("build_patient_timeline_tensors", True)),
            set_all_bloodtest_values_to_one=bool(config.get("set_all_bloodtest_values_to_one", False)),
        )


def validate_removed_rate_of_change(config: Mapping[str, object]) -> None:
    """Allow inactive historical settings without reinterpreting a saved slope model."""
    legacy_bounds = {"rate_of_change_min_days", "rate_of_change_max_days"}.intersection(config)
    if config.get("include_rate_of_change", False) or (
        config.get("include_blood_test_change", False) and legacy_bounds
    ):
        raise ValueError(
            "Rate-of-change features were removed. Use `dataset.include_blood_test_change`, "
            "`dataset.blood_test_change_min_days`, and `dataset.blood_test_change_max_days` "
            "and retrain for signed differences without division by elapsed days."
        )


def validate_blood_test_change_window(min_days: int, max_days: int) -> None:
    if min_days < 1:
        raise ValueError("`dataset.blood_test_change_min_days` must be at least 1.")
    if max_days < min_days:
        raise ValueError("`dataset.blood_test_change_max_days` must be at least blood_test_change_min_days.")


def parse_training_sample_weighting(value: object) -> SampleWeighting:
    return parse_sample_weighting(value, "training_sample_weighting")


def parse_bootstrap_sample_weighting(value: object) -> SampleWeighting:
    return parse_sample_weighting(value, "bootstrap_sample_weighting")


def parse_sample_weighting(value: object, field: str) -> SampleWeighting:
    weighting = str(value)
    if weighting in SAMPLE_WEIGHTING_MODES:
        return cast(SampleWeighting, weighting)
    modes = "`, `".join(SAMPLE_WEIGHTING_MODES)
    raise ValueError(f"`dataset.{field}` must be `{modes}`.")


def parse_selected_blood_tests(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise TypeError("`selected_blood_tests` must be a list of blood-test keys.")
    selected = tuple(str(test_key).strip() for test_key in value)
    if any(test_key == "" for test_key in selected):
        raise ValueError("`selected_blood_tests` cannot contain empty values.")
    if len(set(selected)) != len(selected):
        raise ValueError("`selected_blood_tests` cannot contain duplicate values.")
    return selected


def validate_bc_diagnosis_window_config(
    restrict_bloodtests_to_bc_diagnosis_window: bool,
    bc_diagnosis_days_csv_path: Path | None,
    bc_diagnosis_window_days_before: int,
    bc_diagnosis_window_days_after: int,
) -> None:
    if restrict_bloodtests_to_bc_diagnosis_window and bc_diagnosis_days_csv_path is None:
        raise ValueError(
            "`dataset.bc_diagnosis_days_csv_path` is required when "
            "`dataset.restrict_bloodtests_to_bc_diagnosis_window` is true."
        )
    if bc_diagnosis_window_days_before < 0:
        raise ValueError("`dataset.bc_diagnosis_window_days_before` cannot be negative.")
    if bc_diagnosis_window_days_after < 0:
        raise ValueError("`dataset.bc_diagnosis_window_days_after` cannot be negative.")


def validate_recent_diagnosis_config(
    include_recent_diagnoses: bool,
    diagnosis_procedure_csv_path: Path | None,
    num_most_common_recent_diagnoses: int,
    recent_diagnosis_lookback_days: int,
) -> None:
    require_clinical_event_path(include_recent_diagnoses, diagnosis_procedure_csv_path, "include_recent_diagnoses")
    if num_most_common_recent_diagnoses < 0:
        raise ValueError("`dataset.num_most_common_recent_diagnoses` cannot be negative.")
    if recent_diagnosis_lookback_days < 0:
        raise ValueError("`dataset.recent_diagnosis_lookback_days` cannot be negative.")


def validate_prior_bc_surgery_exclusion_config(
    exclude_prior_bc_surgery_patients: bool,
    diagnosis_procedure_csv_path: Path | None,
    prior_bc_surgery_code_regex: str,
) -> None:
    require_clinical_event_path(
        exclude_prior_bc_surgery_patients,
        diagnosis_procedure_csv_path,
        "exclude_prior_bc_surgery_patients",
    )
    if prior_bc_surgery_code_regex.strip() == "":
        raise ValueError("`dataset.prior_bc_surgery_code_regex` cannot be empty.")
    re.compile(prior_bc_surgery_code_regex)


def validate_other_cancer_config(
    enabled: bool,
    diagnosis_procedure_csv_path: Path | None,
    code_filter_regex: str,
) -> None:
    if not enabled:
        return
    require_clinical_event_path(enabled, diagnosis_procedure_csv_path, "exclude_and_censor_other_cancer_patients")
    if not code_filter_regex:
        raise ValueError(
            "`dataset.other_cancer_code_filter_regex` is required when "
            "`dataset.exclude_and_censor_other_cancer_patients` is true."
        )
    re.compile(code_filter_regex)


def require_clinical_event_path(enabled: bool, path: Path | None, option: str) -> None:
    if enabled and path is None:
        raise ValueError(f"`dataset.diagnosis_procedure_csv_path` is required when `dataset.{option}` is true.")


def validate_landmark_config(min_landmark_day: int) -> None:
    if min_landmark_day < 0:
        raise ValueError("`dataset.min_landmark_day` cannot be negative.")


def validate_min_blood_test_types_per_day(value: int | None) -> None:
    if value is None:
        return
    if value < 1:
        raise ValueError("`dataset.min_blood_test_types_per_day` must be >= 1.")


def validate_min_diagnosis_year(min_diagnosis_year: int | None) -> None:
    if min_diagnosis_year is not None and min_diagnosis_year < 0:
        raise ValueError("`dataset.min_diagnosis_year` cannot be negative.")


def required_path(config: Mapping[str, object], key: str, base_dir: Path) -> Path:
    if key not in config:
        raise KeyError(f"Missing required dataset config key `{key}`.")
    return resolve_path(str(config[key]), base_dir)


def optional_path(value: object, base_dir: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return resolve_path(str(value), base_dir)


def optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def normalize_split(split: str) -> str:
    normalized = split.lower().strip()
    aliases = {"validation": "val", "valid": "val", "dev": "val"}
    return aliases.get(normalized, normalized)
