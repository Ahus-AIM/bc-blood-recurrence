"""Read externally calculated age summaries without consulting binned patient ages."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

SPLIT_KEYS = ('train', 'validation', 'test', 'overall')


@dataclass(frozen=True)
class AgeSummary:
    """Exact-age summary entered by the author: median and ordered quartiles q1/q3, in years."""

    median: float
    q1: float
    q3: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, object], split: str) -> AgeSummary:
        numbers = {key: age_number(values[key], f'{split}.{key}') for key in ('median', 'q1', 'q3')}
        summary = cls(**numbers)
        if not 0 <= summary.q1 <= summary.median <= summary.q3:
            raise ValueError(f'{split}: age quartiles must satisfy 0 ≤ q1 ≤ median ≤ q3.')
        return summary

    def display(self) -> str:
        return f'{display_number(self.median)} ({display_number(self.q1)}–{display_number(self.q3)})'


def age_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{key} must be a finite number; use null for the entire unavailable split.')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{key} must be finite.')
    return number


def display_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


@dataclass(frozen=True)
class ManualAgeSummary:
    """External split summaries with cohort/source provenance; null splits remain unavailable.

    ``age_years`` contains train, validation, test and pooled overall summaries.
    ``cohort_reference`` identifies the cohort and ``source_note`` describes the
    external calculation. These reporting inputs never supply model features.
    """

    age_years: dict[str, AgeSummary | None]
    cohort_reference: str
    source_note: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ManualAgeSummary:
        summaries = raw.get('age_years')
        if not isinstance(summaries, Mapping):
            raise ValueError('Manual ages must contain an age_years mapping.')
        ages: dict[str, AgeSummary | None] = {}
        for split in SPLIT_KEYS:
            value = summaries[split]
            if value is None:
                ages[split] = None
            elif isinstance(value, Mapping):
                ages[split] = AgeSummary.from_mapping(value, split)
            else:
                raise ValueError(f'{split} must be a summary mapping or null.')
        return cls(ages, str(raw['cohort_reference']), str(raw['source_note']))

    @property
    def cells(self) -> list[str]:
        values = (self.age_years[split] for split in SPLIT_KEYS)
        return ['—' if value is None else value.display() for value in values]


def load_manual_age_summary(path: Path) -> ManualAgeSummary:
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, Mapping):
        raise ValueError(f'{path} must contain an age_years mapping.')
    return ManualAgeSummary.from_mapping(raw)


def load_age_cells(path: Path) -> tuple[list[str], dict[str, str]]:
    """Share validated display cells and provenance with the manuscript design scripts."""
    summary = load_manual_age_summary(path)
    return summary.cells, {
        'config_path': str(path),
        'cohort_reference': summary.cohort_reference,
        'source_note': summary.source_note,
    }
