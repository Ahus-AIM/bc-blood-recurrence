from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

from src.model.ebm_explanation_data import (
    float_array as float_array,
    optional_float_array as optional_float_array,
    sequence_value as sequence_value,
)

TERM_COLUMNS = (
    "term_index term_name term_type bin_index bin_label score outer_bag_score_std score_lower score_upper "
    "hazard_ratio hazard_ratio_lower hazard_ratio_upper is_missing"
).split()
INTERACTION_COLUMNS = (
    "term_index term_name left_feature right_feature left_bin_index right_bin_index "
    "left_bin_label right_bin_label score outer_bag_score_std score_lower score_upper "
    "hazard_ratio hazard_ratio_lower hazard_ratio_upper"
).split()


def write_explanation_csvs(
    model: ExplainableBoostingRegressor,
    explanation: Any,
    output_dir: Path,
    include_missing_bins: bool,
) -> dict[str, str]:
    paths = {
        "term_scores_csv": output_dir / "term_scores.csv",
        "interaction_scores_csv": output_dir / "interaction_scores.csv",
    }
    pd.DataFrame(term_rows(model, explanation, include_missing_bins), columns=TERM_COLUMNS).to_csv(
        paths["term_scores_csv"], index=False
    )
    pd.DataFrame(interaction_rows(model, explanation), columns=INTERACTION_COLUMNS).to_csv(
        paths["interaction_scores_csv"], index=False
    )
    return {key: str(path.relative_to(output_dir.parent)) for key, path in paths.items()}


def term_rows(
    model: ExplainableBoostingRegressor,
    explanation: Any,
    include_missing_bins: bool,
) -> Iterator[dict[str, object]]:
    for term_index, term_name in explanation_terms(explanation):
        term_data = explanation.data(term_index)
        if term_data.get("type") == "interaction":
            continue
        yield from visible_term_rows(model, term_data, term_index, term_name)
        if include_missing_bins:
            yield from missing_term_rows(model, term_index, term_name)


def visible_term_rows(
    model: ExplainableBoostingRegressor,
    term_data: Mapping[str, object],
    term_index: int,
    term_name: str,
) -> Iterator[dict[str, object]]:
    scores = float_array(term_data["scores"])
    stds = visible_standard_deviations(model, term_index, scores.shape)
    labels = bin_labels(sequence_value(term_data["names"]), len(scores))
    for bin_index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        yield {
            "term_index": term_index,
            "term_name": term_name,
            "term_type": "univariate",
            "bin_index": bin_index,
            "bin_label": label,
            **score_columns(float(score), std_value(stds, bin_index)),
            "is_missing": False,
        }


def missing_term_rows(
    model: ExplainableBoostingRegressor,
    term_index: int,
    term_name: str,
) -> Iterator[dict[str, object]]:
    term_scores = getattr(model, "term_scores_", ())
    if term_index >= len(term_scores):
        return
    scores = float_array(term_scores[term_index])
    if scores.ndim != 1 or len(scores) == 0:
        return
    stds = full_standard_deviations(model, term_index, scores.shape)
    yield {
        "term_index": term_index,
        "term_name": term_name,
        "term_type": "univariate",
        "bin_index": -1,
        "bin_label": "missing",
        **score_columns(float(scores[0]), std_value(stds, 0)),
        "is_missing": True,
    }


def interaction_rows(model: ExplainableBoostingRegressor, explanation: Any) -> Iterator[dict[str, object]]:
    for term_index, term_name in explanation_terms(explanation):
        term_data = explanation.data(term_index)
        if term_data.get("type") != "interaction":
            continue
        yield from visible_interaction_rows(model, term_data, term_index, term_name)


def visible_interaction_rows(
    model: ExplainableBoostingRegressor,
    term_data: Mapping[str, object],
    term_index: int,
    term_name: str,
) -> Iterator[dict[str, object]]:
    scores = float_array(term_data["scores"])
    stds = visible_standard_deviations(model, term_index, scores.shape)
    left_labels = bin_labels(sequence_value(term_data["left_names"]), scores.shape[0])
    right_labels = bin_labels(sequence_value(term_data["right_names"]), scores.shape[1])
    left_feature, right_feature = interaction_features(term_name)
    for left_index, left_label in enumerate(left_labels):
        for right_index, right_label in enumerate(right_labels):
            yield {
                "term_index": term_index,
                "term_name": term_name,
                "left_feature": left_feature,
                "right_feature": right_feature,
                "left_bin_index": left_index,
                "right_bin_index": right_index,
                "left_bin_label": left_label,
                "right_bin_label": right_label,
                **score_columns(float(scores[left_index, right_index]), std_value(stds, left_index, right_index)),
            }


def score_columns(score: float, score_std: float) -> dict[str, float]:
    score_lower = score - score_std
    score_upper = score + score_std
    return {
        "score": score,
        "outer_bag_score_std": score_std,
        "score_lower": score_lower,
        "score_upper": score_upper,
        "hazard_ratio": hazard_ratio(score),
        "hazard_ratio_lower": hazard_ratio(score_lower),
        "hazard_ratio_upper": hazard_ratio(score_upper),
    }


def hazard_ratio(score: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.exp(score))


def visible_standard_deviations(
    model: ExplainableBoostingRegressor,
    term_index: int,
    score_shape: tuple[int, ...],
) -> np.ndarray | None:
    full_stds = full_standard_deviations(model, term_index, score_shape)
    if full_stds is None or full_stds.shape == score_shape:
        return full_stds
    if full_stds.ndim == 1 and full_stds.shape == (score_shape[0] + 2,):
        return full_stds[1:-1]
    if full_stds.ndim == 2 and full_stds.shape == (score_shape[0] + 2, score_shape[1] + 2):
        return full_stds[1:-1, 1:-1]
    return None


def full_standard_deviations(
    model: ExplainableBoostingRegressor,
    term_index: int,
    score_shape: tuple[int, ...],
) -> np.ndarray | None:
    standard_deviations = getattr(model, "standard_deviations_", None)
    if standard_deviations is None or term_index >= len(standard_deviations):
        return None
    stds = float_array(standard_deviations[term_index])
    return stds if stds.ndim == len(score_shape) else None


def std_value(stds: np.ndarray | None, *index: int) -> float:
    if stds is None:
        return float("nan")
    return float(stds[index])


def bin_labels(names: Sequence[object], score_count: int) -> list[str]:
    numeric = optional_float_array(names)
    if numeric is not None and len(numeric) == score_count + 1:
        return [f"[{numeric[index]:.12g}, {numeric[index + 1]:.12g})" for index in range(score_count)]
    if len(names) >= score_count:
        return [str(name) for name in names[:score_count]]
    return [str(index) for index in range(score_count)]


def interaction_features(term_name: str) -> tuple[str, str]:
    names = term_name.split(" & ", maxsplit=1)
    return (names[0], names[1]) if len(names) == 2 else ("Feature 1", "Feature 2")


def explanation_terms(explanation: Any) -> Iterator[tuple[int, str]]:
    for term_index, term_name in enumerate(sequence_value(explanation.data()["names"])):
        yield term_index, str(term_name)
