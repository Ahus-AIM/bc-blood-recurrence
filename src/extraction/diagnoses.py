"""C50 event export from notebook 04, including its unfiltered source population."""

from __future__ import annotations

import pandas as pd


def extract_diagnosis_days(dips_df: pd.DataFrame) -> pd.DataFrame:
    bc_code_regex = r"C50\d*"
    bc_diagnosis_days = (
        dips_df.loc[
            (dips_df["Datatype"] == "Diagnoser") & dips_df["Kode"].astype("string").str.match(bc_code_regex, na=False),
            ["ID", "Dager_siden_BC_diagnose", "Kode", "Tekst"],
        ]
        .rename(
            columns={
                "ID": "patient_id",
                "Dager_siden_BC_diagnose": "bc_diagnosis_day",
                "Kode": "code",
                "Tekst": "description",
            }
        )
        .copy()
    )
    bc_diagnosis_days["patient_id"] = bc_diagnosis_days["patient_id"].astype(str).str.strip()
    bc_diagnosis_days["bc_diagnosis_day"] = pd.to_numeric(
        bc_diagnosis_days["bc_diagnosis_day"],
        errors="coerce",
    )
    bc_diagnosis_days = bc_diagnosis_days.dropna(subset=["patient_id", "bc_diagnosis_day"])
    bc_diagnosis_days["bc_diagnosis_day"] = bc_diagnosis_days["bc_diagnosis_day"].astype(int)
    bc_diagnosis_days = bc_diagnosis_days.sort_values(
        ["patient_id", "bc_diagnosis_day", "code"],
        kind="mergesort",
    ).reset_index(drop=True)

    return bc_diagnosis_days
