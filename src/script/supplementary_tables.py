"""Export the two supplementary tables that were handwritten in the original manuscript."""

from __future__ import annotations

import shutil

import pandas as pd

from src.config import load_yml
from src.paths import BENCHMARK, CONFIG, PAPER

PARAMETERS = (
    "objective",
    "interactions",
    "max_bins",
    "learning_rate",
    "max_rounds",
    "early_stopping_rounds",
    "min_samples_leaf",
    "outer_bags",
    "validation_size",
    "patient_grouped_outer_bags",
    "patient_grouped_bag_max_attempts",
    "random_state",
    "n_jobs",
)


def write_supplementary_tables() -> None:
    """Report actual saved training settings and copy the manuscript's fixed reference-limit table."""
    raw = load_yml(BENCHMARK / "All/6m/__all_blood_tests__/training_config.yml")
    settings = {"objective": "survival_cox", **raw["ebm"]}
    directory = PAPER / "table"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in PARAMETERS:
        value = settings[name]
        if isinstance(value, bool):
            display = r"\texttt{" + str(value).lower() + "}"
        elif isinstance(value, int):
            display = f"{value:,}"
        elif isinstance(value, str):
            display = r"\texttt{" + value.replace("_", r"\_") + "}"
        else:
            display = str(value)
        if name == "validation_size":
            display += " (fraction of training patients)"
        rows.append(r"\texttt{" + name.replace("_", r"\_") + "} & " + display + r" \\")
    template = (CONFIG / "hyperparameters_table.tex").read_text()
    (directory / "ebm_hyperparameters.tex").write_text(template.replace("%% PARAMETER_ROWS", "\n".join(rows)))
    pd.DataFrame([{"parameter": name, "value": settings[name]} for name in PARAMETERS]).to_csv(
        directory / "ebm_hyperparameters.csv", index=False
    )
    shutil.copy2(CONFIG / "laboratory_reference_table.tex", directory / "laboratory_reference_limits.tex")


if __name__ == "__main__":
    write_supplementary_tables()
