from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.script.plot.dataset_statistics import CONFIG_PATH, build_recurrence_survival_table, load_config
from src.time_units import DAYS_PER_YEAR


@pytest.mark.parametrize(
    "durations, events, threshold, expected_days, expected_percentages",
    [
        pytest.param(
            [10, 20, 30, 40, 50, 60, 70], [True] * 7, 5,
            [0, 10, 20], [100, 600 / 7, 500 / 7], id="hide-death-with-five-at-risk",
        ),
        pytest.param(
            [10, 20, 30, 40, 50, 60, 70], [False, False] + [True] * 5, 5,
            [0], [100], id="censoring-reduces-risk-set-to-five",
        ),
        pytest.param(
            [10, 20, 30, 30, 40, 50, 60], [True] * 7, 5,
            [0, 10, 20], [100, 600 / 7, 500 / 7], id="hide-tied-deaths-with-five-at-risk",
        ),
        pytest.param(
            [10, 20, 30, 40, 50], [True] * 5, 5,
            [], [], id="omit-initial-population-of-five",
        ),
        pytest.param(
            [10] + [365] * 5, [False] * 6, 5,
            [0], [100], id="hide-horizon-with-five-at-risk",
        ),
        pytest.param(
            [10] + [365] * 6, [False] * 7, 5,
            [0, 365], [100, 100], id="retain-supported-horizon",
        ),
        pytest.param(
            [365] * 5 + [400], [True] * 5 + [False], 5,
            [0, 365], [100, 100 / 6], id="use-risk-set-before-tied-deaths-at-horizon",
        ),
        pytest.param(
            [10, 20, 30, 40], [True] * 4, 2,
            [0, 10, 20], [100, 75, 50], id="respect-configured-threshold",
        ),
    ],
)
def test_panel_f_suppresses_steps_at_or_below_risk_threshold(
    durations: list[int],
    events: list[bool],
    threshold: int,
    expected_days: list[int],
    expected_percentages: list[float],
) -> None:
    config = load_config(CONFIG_PATH)
    config = replace(config, first_recurrence_site_survival=replace(
        config.first_recurrence_site_survival,
        max_years_since_origin=1,
        min_patients_at_risk=threshold,
    ))
    endpoint_rows = pd.DataFrame([
        {
            "patient_id": f"P{index}",
            "days_since_diagnosis": 100,
            "code": "C780",
            "death_day": 100 + duration if event else None,
            "last_observation_day": 100 + duration,
        }
        for index, (duration, event) in enumerate(zip(durations, events, strict=True))
    ])

    survival = build_recurrence_survival_table(config, endpoint_rows=endpoint_rows)

    if not expected_days:
        assert survival.empty
        return
    assert set(survival["site"]) == {"Any", "Lung"}
    for _, curve in survival.groupby("site"):
        assert list(curve["years_since_origin"]) == [day / DAYS_PER_YEAR for day in expected_days]
        np.testing.assert_allclose(curve["survival_percent"], expected_percentages, rtol=1e-12, atol=1e-12)
