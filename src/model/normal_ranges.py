from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

FEATURE_KEY_SEPARATOR = "|||"


@dataclass(frozen=True)
class NormalRange:
    """Adult reference range for one blood-test feature.

    `test_name` and `unit` identify the raw blood-test feature, `sex` is
    `female` or `male`, `age_group` records the adult subgroup when Ahus reports
    age-specific intervals, and `lower`/`upper` may be open-ended.
    """

    test_name: str
    unit: str
    sex: str
    age_group: str
    lower: float | None
    upper: float | None
    lower_operator: str
    upper_operator: str
    source_url: str
    notes: str

    @property
    def feature_key(self) -> str:
        return f"{self.test_name}{FEATURE_KEY_SEPARATOR}{self.unit}" if self.unit else self.test_name


def load_normal_ranges(csv_path: Path | None) -> dict[str, tuple[NormalRange, ...]]:
    if csv_path is None:
        return {}
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = tuple(normal_range_from_row(row) for row in csv.DictReader(csv_file))
    return group_ranges_by_feature(rows)


def normal_range_from_row(row: Mapping[str, str]) -> NormalRange:
    return NormalRange(
        test_name=row["test_name"].strip(),
        unit=row["unit"].strip(),
        sex=row["sex"].strip().lower(),
        age_group=row["age_group"].strip(),
        lower=optional_float(row["lower"]),
        upper=optional_float(row["upper"]),
        lower_operator=row["lower_operator"].strip(),
        upper_operator=row["upper_operator"].strip(),
        source_url=row["source_url"].strip(),
        notes=row["notes"].strip(),
    )


def optional_float(value: str) -> float | None:
    stripped = value.strip()
    return None if stripped == "" else float(stripped)


def group_ranges_by_feature(ranges: tuple[NormalRange, ...]) -> dict[str, tuple[NormalRange, ...]]:
    grouped: dict[str, list[NormalRange]] = {}
    for normal_range in ranges:
        grouped.setdefault(normal_range.feature_key, []).append(normal_range)
    return {feature: tuple(values) for feature, values in grouped.items()}


def normal_range_metadata(
    ranges_by_feature: Mapping[str, tuple[NormalRange, ...]],
    csv_path: Path | None,
) -> dict[str, object]:
    return {
        "csv_path": None if csv_path is None else str(csv_path),
        "feature_count": len(ranges_by_feature),
        "range_count": sum(len(ranges) for ranges in ranges_by_feature.values()),
    }
