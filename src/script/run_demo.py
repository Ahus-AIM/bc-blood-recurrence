"""Exercise the study pipeline with marked synthetic inputs and a small fitting budget."""

from __future__ import annotations

import json

import pandas as pd

from src.config import load_yml
from src.paths import BENCHMARK, CONFIG, OUTPUT, PROCESSED, RAW
from src.script import calculate_reference_returns, extract_data
from src.script.generate_synthetic_data import synthetic_frames
from src.script.render_paper import main as render_paper
from src.script.single_bloodtest_benchmark_config import load_config
from src.script.train_single_bloodtest_benchmark import train_from_config


def validate_synthetic_inputs() -> None:
    """Require the exact generated inputs before creating any illustrative age summaries."""
    marker = RAW / "SYNTHETIC.json"
    if not marker.is_file():
        raise ValueError("Generate synthetic data first; this command cannot run on clinical inputs.")
    settings = json.loads(marker.read_text())
    if settings.get("synthetic") is not True:
        raise ValueError("Missing synthetic-data marker.")
    raw, ages, dates = synthetic_frames(int(settings["patients"]), int(settings["seed"]))
    selection = pd.DataFrame({"ID": ages["id"], "to_keep": 1})
    for frame, path in (
        (raw, RAW / "dips.csv"),
        (dates, RAW / "patients.csv"),
        (selection, RAW / "to_keep.csv"),
        (ages, PROCESSED / "bc_diagnosis_patient_age.csv"),
    ):
        if frame.to_csv(index=False).encode() != path.read_bytes():
            raise ValueError(f"Synthetic input changed: {path}. Refusing to create illustrative ages.")


def synthetic_age_returns() -> None:
    """Illustrative summaries only; this is not the pending clinical age calculation."""
    validate_synthetic_inputs()
    membership = pd.read_csv(
        BENCHMARK / "All/6m/__all_blood_tests__/cohort_age_export/cohort_membership.csv", dtype=str
    )
    patients = pd.read_csv(RAW / "patients.csv", dtype=str)
    patients["age"] = (
        pd.to_datetime(patients["day of first bc diagnosis"]).dt.year - pd.to_datetime(patients["birth date"]).dt.year
    )
    joined = membership.merge(patients[["id", "age"]], on="id", validate="many_to_one")
    rows = []
    for fold, frame in joined.groupby("fold", sort=False):
        quantiles = frame["age"].quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "fold": fold,
                "median": quantiles.loc[0.5],
                "lower_q": quantiles.loc[0.25],
                "upper_q": quantiles.loc[0.75],
                "hash": frame["hash"].iloc[0],
            }
        )
    directory = PROCESSED / "ages"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "synthetic.csv", index=False)


def main() -> None:
    validate_synthetic_inputs()
    extract_data.main()
    raw, config_dir = load_config(CONFIG / "ebm_single_bloodtest_benchmark.yml")
    overrides = load_yml(CONFIG / "demo.yml")
    for section, values in overrides.items():
        raw[section].update(values)
    (OUTPUT / "SYNTHETIC.json").write_text(json.dumps({"synthetic": True, "overrides": overrides}, indent=2) + "\n")
    train_from_config(raw, config_dir)
    results = json.loads((BENCHMARK / "metadata.json").read_text())["results"]
    if len(results) != 390 or any(row["status"] != "complete" for row in results):
        raise RuntimeError("Synthetic benchmark did not complete all 130 models across three splits; inspect the log.")
    synthetic_age_returns()
    calculate_reference_returns.main()
    render_paper()


if __name__ == "__main__":
    main()
