from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.data.dataset_config import PatientDatasetConfig
    from src.data.types import PatientBloodTestItem


def validate_temporal_cutoffs(train_end_year: int, test_start_year: int) -> None:
    if train_end_year < 0 or test_start_year <= train_end_year:
        raise ValueError("`dataset.train_end_year` must be nonnegative and less than `dataset.test_start_year`.")


def split_for_diagnosis_year(
    year: int, train_end_year: int, test_start_year: int,
) -> Literal["train", "val", "test"]:
    if year <= train_end_year:
        return "train"
    if year >= test_start_year:
        return "test"
    return "val"


def validate_final_cohort_splits(
    items_by_split: Mapping[str, tuple[PatientBloodTestItem, ...]], config: PatientDatasetConfig,
) -> None:
    """Validate eligible patients without reassigning any source split membership."""
    for split in ("train", "val", "test"):
        for item in items_by_split.get(split, ()):
            expected = split_for_diagnosis_year(item.diagnosis_year, config.train_end_year, config.test_start_year)
            if split != expected:
                raise ValueError(
                    f"Patient {item.patient_id}, diagnosis year {item.diagnosis_year}, "
                    f"is assigned to {split}; expected {expected} "
                    f"(train_end_year={config.train_end_year}, test_start_year={config.test_start_year})."
                )
