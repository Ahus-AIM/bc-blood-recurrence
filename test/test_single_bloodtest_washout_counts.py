from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.regeneration.runner import run_regeneration
from src.script import single_bloodtest_washout
from src.script.ebm_survival_config import write_training_config_snapshot
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY
from src.script.single_bloodtest_washout import prepare_washout_state, washout_landmarks
from src.script.single_bloodtest_washout_counts import build_washout_cohort_counts, write_washout_cohort_outputs
from src.script.survival_benchmark_cache import SurvivalBenchmarkFeatureCache
from test.regeneration_helpers import regeneration_config
from test.test_benchmark_landmarks import every_outcome_dataset
from test.test_patient_bloodtest import blood_row, clinical_row, config, endpoint_row


@pytest.fixture
def count_dataset(tmp_path: Path) -> PatientBloodTestDataset:
    """Include repeated visits, event-day input, horizon ties and competing events."""
    days = {
        "train": (40, 60),
        "liver": (40, 70, 100),
        "lung": (40, 69),
        "late": (40,),
        "short": (40,),
        "long": (40,),
        "other_cancer": (40,),
        "horizon_tie": (40,),
    }
    endpoints = [
        endpoint_row("train", diagnosis_to_ddfs=100, ddfs_code="C787"),
        endpoint_row("liver", diagnosis_to_ddfs=100, ddfs_code="C787"),
        endpoint_row("lung", diagnosis_to_ddfs=100, ddfs_code="C780"),
        endpoint_row("late", diagnosis_to_ddfs=230, ddfs_code="C787"),
        endpoint_row("short", diagnosis_to_death=70),
        endpoint_row("long", last_entry_days=160),
        endpoint_row("other_cancer", diagnosis_to_ddfs=100, ddfs_code="C787"),
        endpoint_row("horizon_tie", diagnosis_to_ddfs=220, ddfs_code="C787"),
    ]
    settings = config(
        tmp_path,
        [blood_row(patient, day) for patient, visits in days.items() for day in visits],
        endpoints,
        split_by_patient={patient: "train" if patient == "train" else "test" for patient in days},
        clinical_rows=[clinical_row("other_cancer", 100, code="C80")],
    )
    return PatientBloodTestDataset({
        **settings,
        "min_landmark_day": 31,
        "include_recurrence_day_blood_tests": True,
        "exclude_and_censor_other_cancer_patients": True,
        "other_cancer_code_filter_regex": "C80",
        "test_bootstrap_samples": 1372,
        "selected_blood_tests": ["B-Hb|||g/dL"],
    })


@pytest.fixture
def count_pools(count_dataset: PatientBloodTestDataset) -> dict[int, SurvivalBenchmarkFeatureCache]:
    return washout_landmarks(count_dataset, (0, 30, 60, 90, 180, 360))


def count_benchmark() -> dict[str, object]:
    return {
        "sites": [
            {"label": "All", "endpoint_ddfs_code_filter_regex": r"C7[89]\d*"},
            {"label": "Liver", "endpoint_ddfs_code_filter_regex": "C787"},
            {"label": "Brain", "endpoint_ddfs_code_filter_regex": "C793"},
        ],
        "horizons": [{"label": f"{days}d", "days": days} for days in (30, 180, 360, 720)],
    }


@pytest.mark.parametrize(("site", "horizon", "washout", "expected"), [
    ("All", 180, 0, (7, 10, 3, 6, 3)),
    ("Liver", 180, 0, (7, 10, 2, 4, 5)),
    ("Brain", 180, 0, (7, 10, 0, 0, 8)),
    ("All", 30, 0, (7, 10, 1, 2, 0)),
    ("All", 360, 0, (7, 10, 4, 7, 3)),
    ("All", 180, 30, (6, 7, 3, 4, 2)),
    ("Liver", 180, 30, (6, 7, 2, 2, 4)),
    ("All", 180, 60, (3, 3, 1, 1, 1)),
    ("All", 180, 90, (3, 3, 1, 1, 1)),
    ("All", 360, 180, (1, 1, 1, 1, 0)),
    ("All", 720, 360, (0, 0, 0, 0, 0)),
    ("All", 180, 180, (None, None, None, None, None)),
    ("All", 180, 360, (None, None, None, None, None)),
])
def test_counts_use_unique_test_patients_and_endpoint_specific_horizon_events(
    count_pools: dict[int, SurvivalBenchmarkFeatureCache],
    site: str,
    horizon: int,
    washout: int,
    expected: tuple[int | None, int | None, int | None, int | None, int | None],
) -> None:
    rows = build_washout_cohort_counts(count_benchmark(), count_pools)
    row = next(row for row in rows if (row.site_label, row.horizon_days, row.washout_days) == (site, horizon, washout))

    assert (row.patient_count, row.landmark_count, row.event_patient_count,
            row.event_landmark_count, row.early_censored_landmark_count) == expected
    assert row.status == ("skipped" if washout >= horizon else "complete")
    assert row.split == "test"


@pytest.mark.parametrize(("horizon", "washout", "landmarks", "censored"), [
    (29, 0, 15, (0, 0, 0)),
    (30, 0, 15, (3, 4, 5)),
    (31, 0, 15, (6, 8, 10)),
    (180, 0, 15, (9, 12, 15)),
    (30, 29, 10, (0, 0, 0)),
    (31, 29, 10, (3, 4, 5)),
    (180, 30, 5, (3, 4, 5)),
])
def test_censoring_counts_all_model_causes_strictly_before_the_horizon(
    tmp_path: Path,
    horizon: int,
    washout: int,
    landmarks: int,
    censored: tuple[int, int, int],
) -> None:
    # Each of five outcomes has landmarks 29, 30 and 31 days before it.
    # Death, last observation and other cancer censor every endpoint; recurrence
    # at another site also censors site-specific models. Horizon ties do not count.
    dataset = every_outcome_dataset(tmp_path)
    benchmark = {**count_benchmark(), "horizons": [{"label": "test horizon", "days": horizon}]}
    rows = build_washout_cohort_counts(benchmark, washout_landmarks(dataset, (0, washout)))
    selected = [row for row in rows if row.washout_days == washout]

    assert [row.landmark_count for row in selected] == [landmarks] * 3
    assert tuple(row.early_censored_landmark_count for row in selected) == censored


def test_header_totals_include_late_recurrences_and_survive_washout(
    count_pools: dict[int, SurvivalBenchmarkFeatureCache],
) -> None:
    rows = build_washout_cohort_counts(count_benchmark(), count_pools)
    expected = {"All": 4, "Liver": 3, "Brain": 0}
    for row in rows:
        assert row.recurrence_patient_count == expected[row.site_label]
    # The late recurrence contributes to the header, but no 180-day event landmark.
    unwashed = next(row for row in rows if (row.site_label, row.horizon_days, row.washout_days) == ("All", 180, 0))
    assert unwashed.event_patient_count == 3
    # Even after washout removes every landmark, the original recurrence total remains.
    empty = next(row for row in rows if (row.site_label, row.horizon_days, row.washout_days) == ("All", 720, 360))
    assert empty.patient_count == 0
    assert empty.recurrence_patient_count == 4


@pytest.mark.parametrize(("days", "codes", "death", "other_cancer", "expected"), [
    ([100, 100], ["C78.7", "C780"], None, None, (1, 1, 1)),
    ([100, 140], ["C780", "C787"], None, None, (1, 0, 1)),
    ([100], ["C787"], None, 100, (0, 0, 0)),
    ([120], ["C787"], 100, None, (1, 1, 0)),
])
def test_header_totals_use_first_sites_and_shared_outcome_precedence(
    tmp_path: Path,
    days: list[int],
    codes: list[str],
    death: int | None,
    other_cancer: int | None,
    expected: tuple[int, int, int],
) -> None:
    settings = config(
        tmp_path,
        [blood_row(patient, 40) for patient in ("train", "test")],
        [endpoint_row("train", diagnosis_to_ddfs=100, ddfs_code="C787"), endpoint_row(
            "test", diagnosis_to_ddfs=json.dumps(days), ddfs_code=json.dumps(codes),
            diagnosis_to_death="" if death is None else death,
        )],
        split_by_patient={"train": "train", "test": "test"},
        clinical_rows=[clinical_row("test", 40)] if other_cancer is None else [
            clinical_row("test", other_cancer, code="C80"),
        ],
    )
    dataset = PatientBloodTestDataset({
        **settings, "selected_blood_tests": ["B-Hb|||g/dL"], "min_landmark_day": 31,
        "exclude_and_censor_other_cancer_patients": True, "other_cancer_code_filter_regex": "C80",
    })
    benchmark = {
        "sites": [
            {"label": label, "endpoint_ddfs_code_filter_regex": pattern}
            for label, pattern in (("All", r"C7[89]\d*"), ("Liver", "C787"), ("Lung", "C780"))
        ],
        "horizons": [{"label": "6m", "days": 180}],
    }
    rows = build_washout_cohort_counts(benchmark, washout_landmarks(dataset, (0, 90, 180)))
    expected_by_site = dict(zip(("All", "Liver", "Lung"), expected, strict=True))
    assert all(row.recurrence_patient_count == expected_by_site[row.site_label] for row in rows)
    assert all(row.early_censored_landmark_count == 1 - expected_by_site[row.site_label]
               for row in rows if row.washout_days == 0)


def test_outputs_preserve_counts_nulls_order_and_month_units(
    tmp_path: Path,
    count_pools: dict[int, SurvivalBenchmarkFeatureCache],
) -> None:
    benchmark = {
        "sites": [{"label": "Liver & lung", "endpoint_ddfs_code_filter_regex": "C78[07]"}],
        "horizons": [{"label": "6m", "days": 180}, {"label": "24m", "days": 720}],
        "washout_cohort_counts": {
            "output_csv_path": "tables/counts.csv",
            "output_latex_path": "tables/counts.tex",
            "output_metadata_path": "tables/counts.json",
            "table_label": "tab:custom_counts",
        },
    }

    paths = write_washout_cohort_outputs(benchmark, count_pools, tmp_path)

    frame = pd.read_csv(paths["csv"])
    metadata = json.loads(Path(paths["metadata"]).read_text())
    latex = Path(paths["latex"]).read_text()
    assert frame["washout_days"].tolist() == [0, 30, 60, 90, 180, 360] * 2
    assert frame["horizon_days"].tolist() == [180] * 6 + [720] * 6
    assert frame.iloc[0]["patient_count"] == 7
    assert frame.iloc[0]["landmark_count"] == 10
    assert frame.iloc[0]["event_patient_count"] == 3
    assert frame.iloc[0]["event_landmark_count"] == 6
    assert pd.isna(frame.iloc[4]["patient_count"])
    assert metadata["results"][4]["patient_count"] is None
    assert metadata["results"][-1]["patient_count"] == 0
    assert metadata["format_version"] == 4
    assert frame.iloc[0]["early_censored_landmark_count"] == 3
    assert pd.isna(frame.iloc[4]["early_censored_landmark_count"])
    assert metadata["results"][4]["early_censored_landmark_count"] is None
    assert metadata["results"][-1]["early_censored_landmark_count"] == 0
    assert frame["recurrence_patient_count"].tolist() == [4] * 12
    assert all(row["recurrence_patient_count"] == 4 for row in metadata["results"])
    assert metadata["results"][4]["event_landmark_count"] is None
    assert r"\label{tab:custom_counts}" in latex
    assert r"Measure & Washout interval & Liver \& lung \\" in latex
    assert r"\multicolumn{3}{l}{Prediction horizon: 6 months}" in latex
    assert r"\multicolumn{3}{l}{Prediction horizon: 24 months}" in latex
    assert r"\multirow[c]{6}{*}{Patients} & None & 3 / 7 (42.9\%)" in latex
    assert r" & 1 month & 3 / 6 (50.0\%)" in latex
    assert (
        r"\multirow[c]{6}{*}{\shortstack[l]{Landmarks with recurrence\\within 6 months}}"
        r" & None & 6 / 10 (60.0\%)" in latex
    )
    assert r"\shortstack[l]{Landmarks with recurrence\\within 24 months}" in latex
    assert r"\multirow[c]{6}{*}{Censored landmarks} & None & 3 / 10 (30.0\%)" in latex
    assert r" & 1 month & 2 / 7 (28.6\%)" in latex
    assert r" & 1 month & 4 / 7 (57.1\%)" in latex
    assert r" & 3 months & 1 / 3 (33.3\%)" in latex
    assert " & 6 months & --" in latex
    assert r" & 6 months & 1 / 1 (100.0\%)" in latex
    assert " & 12 months & 0 / 0 (--)" in latex
    assert r"\begin{table*}[htbp]" in latex
    assert "sidewaystable" not in latex
    assert r"\textheight" not in latex
    assert "before bootstrap sampling" in latex
    assert "patient_id" not in frame.columns


def test_count_regeneration_exports_and_bundles_without_models_or_bootstrap_draws(
    tmp_path: Path,
    count_dataset: PatientBloodTestDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = {
        **count_benchmark(),
        "horizons": [{"label": "6m", "days": 180}],
        "washout": {"days": [0, 30, 60, 90]},
    }
    source = {
        "site_label": "All", "horizon_label": "6m", "horizon_days": 180,
        "model_key": ALL_MODEL_KEY, "split": "test", "status": "complete",
    }
    (tmp_path / "metadata.json").write_text(json.dumps({
        "selected_blood_tests": count_dataset.selected_test_keys, "results": [source],
    }))
    saved = tmp_path / "All/6m" / ALL_MODEL_KEY
    saved.mkdir(parents=True)
    write_training_config_snapshot({}, count_dataset.config, saved, tmp_path)
    path = regeneration_config(tmp_path, benchmark, ["washout_counts", "paper_bundle"])

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Counts must not load models or sample bootstrap datasets")

    monkeypatch.setattr("src.regeneration.context.load_model", forbidden)
    monkeypatch.setattr(BenchmarkLandmarks, "bootstrap_indices", forbidden)
    run_regeneration(path)

    latex = (tmp_path / "paper_bundle/table/cohort_counts.tex").read_text()
    assert "Prediction horizon: 6 months." in latex
    for measure in (
        "Patients", r"\shortstack[l]{Landmarks with recurrence\\within 6 months}", "Censored landmarks",
    ):
        assert latex.count(rf"\multirow[c]{{4}}{{*}}{{{measure}}}") == 1
    assert r"Measure & Washout interval & Any & Liver & Brain \\" in latex
    assert r"Patients} & None & 3 / 7 (42.9\%) & 2 / 7 (28.6\%) & 0 / 7 (0.0\%)" in latex
    assert (
        r"Landmarks with recurrence\\within 6 months}}"
        r" & None & 6 / 10 (60.0\%) & 4 / 10 (40.0\%) & 0 / 10 (0.0\%)" in latex
    )
    assert r"Censored landmarks} & None & 3 / 10 (30.0\%) & 5 / 10 (50.0\%) & 8 / 10 (80.0\%)" in latex
    assert r" & 1 month & 2 / 7 (28.6\%) & 4 / 7 (57.1\%) & 5 / 7 (71.4\%)" in latex
    assert r" & 1 month & 3 / 6 (50.0\%) & 2 / 6 (33.3\%) & 0 / 6 (0.0\%)" in latex
    assert r" & 1 month & 4 / 7 (57.1\%) & 2 / 7 (28.6\%) & 0 / 7 (0.0\%)" in latex
    frame = pd.read_csv(tmp_path / "paper_bundle/table/cohort_counts.csv")
    assert frame["site_label"].tolist() == ["All"] * 4 + ["Liver"] * 4 + ["Brain"] * 4
    assert frame.loc[frame.site_label == "Brain", "event_landmark_count"].tolist() == [0] * 4
    assert frame.loc[frame.site_label == "Brain", "early_censored_landmark_count"].tolist() == [8, 5, 1, 1]
    assert not list(tmp_path.rglob("*.pkl"))


def test_count_export_failure_keeps_washout_evaluation_available(
    tmp_path: Path,
    count_dataset: PatientBloodTestDataset,
    count_pools: dict[int, SurvivalBenchmarkFeatureCache],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_export(*_args: object) -> None:
        raise OSError("report path is not writable")

    monkeypatch.setattr(single_bloodtest_washout, "write_washout_cohort_outputs", fail_export)
    state = prepare_washout_state(count_dataset, count_benchmark(), count_pools[0], tmp_path)

    assert state is not None
    assert state[0][0] is count_pools[0]
    assert tuple(state[0]) == (0, 30, 60, 90)
    assert "washout cohort-count output failed" in caplog.text
