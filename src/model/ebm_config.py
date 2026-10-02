from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from interpret.glassbox import ExplainableBoostingRegressor

EBM_DEFAULTS = ExplainableBoostingRegressor().get_params()
GROUPED_BAG_OPTIONS = {"patient_grouped_outer_bags", "patient_grouped_bag_max_attempts"}


def resolve_ebm_config(config: Mapping[str, object]) -> dict[str, object]:
    """Resolve native defaults and overrides while retaining survival input semantics."""
    unknown = set(config).difference(EBM_DEFAULTS, GROUPED_BAG_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown EBM parameters: {sorted(unknown)}")
    if "feature_names" in config and config["feature_names"] is not None:
        raise ValueError("EBM feature_names are supplied by the configured dataset.")
    if config.get("objective", "survival_cox") != "survival_cox":
        raise ValueError("EBM objective must be survival_cox.")
    parameters = {**deepcopy(EBM_DEFAULTS), **deepcopy(dict(config)), "objective": "survival_cox"}
    parameters.pop("feature_names")
    return parameters


def ebm_constructor_parameters(config: Mapping[str, object]) -> dict[str, object]:
    """Separate native constructor options from project-specific bagging options."""
    return {key: value for key, value in resolve_ebm_config(config).items() if key not in GROUPED_BAG_OPTIONS}
