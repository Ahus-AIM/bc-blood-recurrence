from __future__ import annotations

import numpy as np
import pytest

from src.data.recurrence_site_incidence import kaplan_meier_step_points
from src.time_units import DAYS_PER_YEAR


@pytest.mark.parametrize(
    "durations, events, minimum, expected_days, expected_percentages",
    [
        pytest.param(
            [10, 10, 20, 365], [True, False, True, False], 1,
            [0, 10, 20, 365], [100, 75, 37.5, 37.5], id="tied-death-and-censoring",
        ),
        pytest.param(
            [10, 10, 365], [True, True, False], 1,
            [0, 10, 365], [100, 100 / 3, 100 / 3], id="tied-deaths",
        ),
        pytest.param(
            [0, 0, 365], [True, False, False], 1,
            [0, 0, 365], [100, 200 / 3, 200 / 3], id="zero-day-death-keeps-initial-point",
        ),
        pytest.param(
            [365, 365, 500], [True, False, True], 1,
            [0, 365], [100, 200 / 3], id="death-at-horizon-is-included-once",
        ),
        pytest.param(
            [400, 500], [True, True], 1,
            [0, 365], [100, 100], id="deaths-beyond-horizon",
        ),
        pytest.param(
            [10, 365], [False, False], 1,
            [0, 365], [100, 100], id="all-censored-supported-tail",
        ),
        pytest.param(
            [10, 20], [False, False], 1,
            [0], [100], id="all-censored-unsupported-tail",
        ),
        pytest.param(
            [10, 20], [True, True], 1,
            [0, 10, 20], [100, 50, 0], id="all-die-no-extrapolation",
        ),
        pytest.param(
            [10, 20, 20, 365], [True, True, False, False], 3,
            [0, 10, 20], [100, 75, 50], id="risk-set-exactly-at-threshold",
        ),
        pytest.param(
            [10, 20, 30, 365], [True, False, True, False], 3,
            [0, 10], [100, 75], id="suppress-death-after-censoring-reduces-risk-set",
        ),
        pytest.param(
            [10, 365], [True, False], 3,
            [], [], id="initial-population-below-threshold",
        ),
        pytest.param([], [], 1, [], [], id="empty-population"),
        pytest.param(
            [np.nan, np.inf, -np.inf, 10, 365], [True, True, False, True, False], 1,
            [0, 10, 365], [100, 50, 50], id="nonfinite-durations-excluded-with-aligned-events",
        ),
        pytest.param(
            [np.nan, np.inf], [True, False], 1,
            [], [], id="no-finite-durations",
        ),
    ],
)
def test_kaplan_meier_preserves_event_steps_and_anonymity_rules(
    durations: list[float],
    events: list[bool],
    minimum: int,
    expected_days: list[int],
    expected_percentages: list[float],
) -> None:
    points = kaplan_meier_step_points(
        np.asarray(durations, dtype=float), np.asarray(events, dtype=bool),
        max_years_since_origin=1, min_patients_at_risk=minimum,
    )

    assert [year for year, _ in points] == [day / DAYS_PER_YEAR for day in expected_days]
    np.testing.assert_allclose([percent for _, percent in points], expected_percentages, rtol=1e-12, atol=1e-12)
