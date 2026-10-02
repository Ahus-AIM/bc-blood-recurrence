from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib.metadata import version

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

from src.data import PatientLandmarkTable
from src.data.timing import timed
from src.model.ebm_bags import (
    PatientGroupedBagPlan,
    build_patient_grouped_bag_plan,
    patient_grouped_bags_enabled,
)
from src.model.ebm_config import ebm_constructor_parameters

LOGGER = logging.getLogger(__name__)


@timed(LOGGER, "fit EBM survival model")
def fit_ebm_model(
    train_table: PatientLandmarkTable,
    train_features: np.ndarray,
    config: Mapping[str, object],
    *,
    patient_grouped_bag_plan: PatientGroupedBagPlan | None = None,
) -> tuple[ExplainableBoostingRegressor, dict[str, object]]:
    """Fit EBM Cox with configured landmark weights and patient-grouped bags."""
    parameters = ebm_constructor_parameters(config)
    bag_plan = patient_grouped_bag_plan
    if patient_grouped_bags_enabled(config):
        if bag_plan is None:
            bag_plan = build_patient_grouped_bag_plan(train_table, config)
        if bag_plan.bags.shape[0] != len(train_table.patient_ids):
            raise ValueError("Patient-grouped EBM bag rows must match training landmark rows.")
        LOGGER.info(
            "using patient-grouped EBM outer bags: outer_bags=%d validation_patients=%d total_redraws=%d",
            bag_plan.metadata["outer_bags"],
            bag_plan.metadata["validation_patient_count"],
            bag_plan.metadata["total_redraws"],
        )
    elif bag_plan is not None:
        raise ValueError("A patient-grouped EBM bag plan was supplied while grouped bags are disabled.")
    model = ExplainableBoostingRegressor(
        feature_names=list(train_table.feature_names),
        **parameters,
    )
    model.fit(
        train_features,
        train_table.signed_durations,
        sample_weight=train_table.sample_weights,
        bags=None if bag_plan is None else bag_plan.bags,
    )
    metadata: dict[str, object] = {
        "library": "interpret",
        "version": version("interpret"),
        "config": dict(config),
        "effective_parameters": parameters,
        "patient_grouped_outer_bags_enabled": bag_plan is not None,
    }
    if bag_plan is not None:
        metadata["patient_grouped_outer_bags"] = bag_plan.metadata
    return model, metadata
