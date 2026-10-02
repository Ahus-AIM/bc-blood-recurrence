from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch

from src.data.dataset_config import PatientDatasetConfig

if TYPE_CHECKING:
    from src.data.cohort_comparison_data import CohortComparisonPatient
    from src.data.recurrence_measurement_history import RecurrenceMeasurementHistory

RightCensoringReason = Literal[
    "death",
    "last_observation",
    "non_endpoint_recurrence",
    "other_cancer",
]


@dataclass(frozen=True)
class PatientEndpointEvent:
    """Distant recurrence event with patient-relative day, code, and optional description."""

    days_since_diagnosis: int
    code: str
    description: str = ""


@dataclass(frozen=True)
class PatientClinicalEvent:
    """Diagnosis/procedure event for timeline plots, with day, type, code, and description."""

    days_since_diagnosis: int
    event_type: str
    code: str = ""
    description: str = ""


@dataclass(frozen=True)
class PatientRecord:
    """Endpoint CSV patient row with diagnosis timing, outcomes, and distant recurrence events."""

    patient_id: str
    diagnosis_year: int
    death_day: int | None
    last_observation_day: int | None
    recurrence_events: tuple[PatientEndpointEvent, ...]


@dataclass(frozen=True)
class PatientDataSources:
    """Loaded endpoint, blood-test, split, age, clinical-event, and drop-count sources.

    ``blood_df`` contains full-history numeric measurements after patient/day/type
    deduplication, shared by model preparation and descriptive blood-test counts.
    """

    endpoint_patients: dict[str, PatientRecord]
    blood_df: pd.DataFrame
    split_by_patient: dict[str, str]
    age_by_patient: dict[str, float]
    clinical_events_by_patient: dict[str, tuple[PatientClinicalEvent, ...]]
    bc_diagnosis_events_by_patient: dict[str, tuple[PatientClinicalEvent, ...]]
    dropped_bloodtest_counts: dict[str, int]
    dropped_clinical_event_counts: dict[str, int]


@dataclass(frozen=True)
class PatientEndpointStatus:
    """Resolved recurrence, discard, and unified right-censoring status."""

    input_recurrence_event: PatientEndpointEvent | None
    endpoint_event: PatientEndpointEvent | None
    non_endpoint_events: tuple[PatientEndpointEvent, ...]
    discard_reason: str | None
    right_censoring_day: int | None = None
    right_censoring_reason: RightCensoringReason | None = None


@dataclass(frozen=True)
class PatientMeasurementSummary:
    """Compact selected-measurement days for one patient.

    `post_diagnosis_days` retains its historical name but includes all sorted
    retained days, including enabled pre-diagnosis history.
    `test_counts_by_day` maps each such day to the number of distinct
    selected blood-test types measured that day. `max_day` is the latest
    selected blood-test day for timeline display.
    """

    post_diagnosis_days: tuple[int, ...]
    test_counts_by_day: Mapping[int, int]
    max_day: int | None


@dataclass(frozen=True)
class PatientMeasurementViews:
    """Selected measurements keyed by patient id, either as full groups or compact day summaries."""

    groups: dict[str, pd.DataFrame]
    summaries: dict[str, PatientMeasurementSummary]


@dataclass(frozen=True)
class PatientCohortStage:
    """One sequential cohort stage and its retained patients and blood-test rows.

    `key` identifies the applied rule, `patient_ids` contains patients with at
    least one landmark candidate after the minimum-day stage, and `bloodtest_count`
    counts deduplicated selected blood-test rows retained through the stage.
    """

    key: str
    patient_ids: frozenset[str]
    bloodtest_count: int


@dataclass(frozen=True)
class ItemBuildContext:
    """Config, loaded sources, endpoint status, measurements, and test ids used to build items."""

    config: PatientDatasetConfig
    sources: PatientDataSources
    status_by_patient: dict[str, PatientEndpointStatus]
    measurements: PatientMeasurementViews
    test_key_to_id: dict[str, int]


@dataclass(frozen=True)
class PatientBloodTestItem:
    """Patient metadata, blood-test tensors, outcome days, and valid landmarks.

    ``right_censoring_day`` and ``right_censoring_reason`` store the resolved
    model boundary. ``source_last_observation_day`` preserves administrative
    follow-up from the endpoint source, including None for a missing source day.
    ``last_observation_day`` is only the display extent for known timeline events.
    """

    patient_id: str
    split: str
    diagnosis_year: int
    age_at_diagnosis: float
    blood_days_since_diagnosis: torch.Tensor
    blood_test_type_ids: torch.Tensor
    blood_values: torch.Tensor
    input_recurrence_day: int | None
    endpoint_day: int | None
    death_day: int | None
    right_censoring_day: int | None
    last_observation_day: int
    source_last_observation_day: int | None
    distant_recurrence_events: tuple[PatientEndpointEvent, ...]
    non_endpoint_recurrence_events: tuple[PatientEndpointEvent, ...] = ()
    bc_diagnosis_events: tuple[PatientClinicalEvent, ...] = ()
    clinical_events: tuple[PatientClinicalEvent, ...] = ()
    discard_reasons: tuple[str, ...] = ()
    valid_landmark_days: tuple[int, ...] = ()
    right_censoring_reason: RightCensoringReason | None = None


@dataclass(frozen=True)
class BloodTestChangeInputs:
    """Observed past values for each landmark row and selected test.

    ``test_keys`` orders the columns of ``past_values``. Each cell holds that
    test's latest eligible observation, independently of whether the current
    value exists. Missing past observations are NaN. These values are imputed
    before deriving model changes, and never receive LOCF.
    """

    test_keys: tuple[str, ...]
    past_values: np.ndarray

    def select_rows(self, indices: np.ndarray) -> BloodTestChangeInputs:
        return BloodTestChangeInputs(self.test_keys, self.past_values[indices])


@dataclass(frozen=True)
class PatientLandmarkTable:
    """Model features, survival targets, landmark identities, training weights, and run metadata.

    ``durations`` preserves recorded follow-up, including zero days.
    ``signed_durations`` encodes zero days as 1e-6, then adds 1e-6 to censored
    durations before negating them for model fitting. Same-day events therefore
    precede censorings; zero-day targets are +1e-6 for events and -2e-6 for censorings.
    ``sample_weight_interval_days`` records forward coverage used by
    person-time-weighted training, counting the event or censoring day once when enabled.
    It is absent when that mode is inactive.
    ``bootstrap_selection_probabilities`` records each drawn landmark's full
    configured selection probability and is absent otherwise.
    ``target_end_reasons`` explains each row's event or censoring boundary.
    ``blood_test_change_inputs`` retains past observations separately from
    the raw differences in ``features`` for imputation before subtraction.
    """

    features: np.ndarray
    signed_durations: np.ndarray
    durations: np.ndarray
    events: np.ndarray
    sample_weights: np.ndarray | None
    feature_names: tuple[str, ...]
    patient_ids: np.ndarray
    landmark_days: np.ndarray
    prior_non_endpoint_recurrence: np.ndarray
    metadata: dict[str, object]
    sample_weight_interval_days: np.ndarray | None = None
    bootstrap_selection_probabilities: np.ndarray | None = None
    target_end_reasons: np.ndarray | None = None
    right_censoring_days: np.ndarray | None = None
    right_censoring_reasons: np.ndarray | None = None
    blood_test_change_inputs: BloodTestChangeInputs | None = None


@dataclass(frozen=True)
class PreparedPatientData:
    """Selected tests, split items, model measurements, display exclusions, and metadata.

    ``measurement_df`` contains retained landmark measurements and optional
    earlier observed history only when change features are enabled.
    ``recurrence_history`` references early and sparse study rows for reports,
    without retaining their values in the model dataset.
    ``excluded_measurement_df`` contains display-only rows removed by Panel A's
    measurement stages for patients who remain in the final cohort.
    ``cohort_comparison_patients`` retains compact full-source histories for the
    supplementary included/excluded report, independently of model filtering.
    """

    selected_test_keys: tuple[str, ...]
    selected_recent_diagnosis_codes: tuple[str, ...]
    selected_recent_diagnosis_feature_names: tuple[str, ...]
    recent_diagnosis_descriptions: dict[str, str]
    recent_diagnosis_train_patient_counts: dict[str, int]
    status_by_patient: dict[str, PatientEndpointStatus]
    items_by_split: dict[str, tuple[PatientBloodTestItem, ...]]
    discarded_patient_items_by_id: dict[str, PatientBloodTestItem]
    measurement_df: pd.DataFrame
    excluded_measurement_df: pd.DataFrame
    dropped_bloodtest_counts: dict[str, int]
    dropped_clinical_event_counts: dict[str, int]
    bc_diagnosis_window_filter_counts: dict[str, object]
    cohort_stages: tuple[PatientCohortStage, ...]
    input_fingerprint: str
    recurrence_history: RecurrenceMeasurementHistory | None = None
    cohort_comparison_patients: tuple[CohortComparisonPatient, ...] = ()
