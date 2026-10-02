from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from src.model.ebm_bags import build_patient_grouped_bag_plan, patient_grouped_bags_enabled
from src.script.survival_training import PreparedSurvivalTraining


def single_bloodtest_task_training(
    prepared: PreparedSurvivalTraining | None,
    not_estimable_reason: str | None,
    ebm_config: Mapping[str, object],
) -> PreparedSurvivalTraining | None:
    """Attach one Endpoint-Definition-valid EBM bag plan to every feature model."""
    if prepared is None or not_estimable_reason is not None:
        return prepared
    if not patient_grouped_bags_enabled(ebm_config):
        return prepared
    bag_plan = build_patient_grouped_bag_plan(prepared.train_table, ebm_config)
    return replace(prepared, patient_grouped_bag_plan=bag_plan)
