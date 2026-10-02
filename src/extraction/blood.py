"""Blood and clinical exports; notebook 02 transformations retain their original order."""

from __future__ import annotations

import pandas as pd


def extract_blood_tests(source: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    """Prefer numeric Svar1, normalize names/units, and retain the notebook's row order."""
    df = source.copy()
    patients_df = endpoints
    df.loc[:, "Svar1"] = (
        df.loc[:, "Svar1"]
        .astype("string")
        .str.strip()
        .str.replace(r"^<\s*(?=\d)", "", regex=True)
        .str.replace(r"^>\s*(?=\d)", "", regex=True)
        .str.replace(r"^(\d+),(\d+)$", r"\1.\2", regex=True)
    )

    df.loc[:, "Svar2"] = (
        df.loc[:, "Svar2"]
        .astype("string")
        .str.strip()
        .str.replace(r"^<\s*(?=\d)", "", regex=True)
        .str.replace(r"^>\s*(?=\d)", "", regex=True)
        .str.replace(r"^(\d+),(\d+)$", r"\1.\2", regex=True)
    )
    df["Svar1"] = df["Svar1"].astype(float, errors="ignore")
    df["Svar2"] = df["Svar2"].astype(float, errors="ignore")
    bt_df = df[~(df["Svar1"].isna() & df["Svar2"].isna()) & df["ID"].isin(patients_df["id"])]

    return normalize_blood_tests(bt_df, patients_df)


def normalize_blood_tests(bt_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    """Preserve sequential name substitutions and the original HbA1c conversion."""
    out_df = bt_df[bt_df["ID"].isin(patients_df["id"])].reset_index(drop=True)
    out_df["Tekst"] = out_df["Tekst"].astype(str)

    out_df["Tekst"] = out_df["Tekst"].str.replace(r"KOS_FLAB_", r"", regex=True)
    out_df["Tekst"] = out_df["Tekst"].str.replace(r"KOS_HKLIN_", r"", regex=True)
    out_df["Tekst"] = out_df["Tekst"].str.replace(r"KOS_KKLIN_", r"", regex=True)
    out_df["Tekst"] = out_df["Tekst"].str.replace(r"KOS_EKLIN_", r"", regex=True)
    out_df["Tekst"] = out_df["Tekst"].str.replace(r"KOS_GKLIN_", r"", regex=True)
    out_df["Tekst"] = out_df["Tekst"].str.replace(r"KOS_LKLIN_", r"", regex=True)

    out_df["Tekst"] = out_df["Tekst"].str.replace(r"B-Lymfocytt.", r"B-Lymfocytter", regex=True)
    out_df["Tekst"] = out_df["Tekst"].str.replace(r"B-Lymfocytterr", r"B-Lymfocytter", regex=True)
    out_df["Tekst"] = out_df["Tekst"].str.replace(r"B-Hemoglobin \(Hb\)", r"B-Hemoglobin", regex=True)

    out_df["Tekst"] = out_df["Tekst"].str.replace(r"P-Troponin T \(serumprøve\)", r"P-Troponin T", regex=True)
    out_df["Tekst"] = out_df["Tekst"].str.replace(r"P-NT-proBNP \(serumprøve\)", r"P-NT-proBNP", regex=True)

    out_df["Prøvesvar_ENHET"] = out_df["Prøvesvar_ENHET"].str.lower()
    out_df["Prøvesvar_ENHET"] = out_df["Prøvesvar_ENHET"].str.replace("10*3/l", "x10e3/l", regex=False)
    out_df["Prøvesvar_ENHET"] = out_df["Prøvesvar_ENHET"].str.replace("10*6/l", "x10e6/l", regex=False)
    out_df["Prøvesvar_ENHET"] = out_df["Prøvesvar_ENHET"].str.replace("10*9/l", "x10e9/l", regex=False)
    out_df["Prøvesvar_ENHET"] = out_df["Prøvesvar_ENHET"].str.replace("10*12/l", "x10e12/l", regex=False)

    # NOTE: Throwing all non-float values.
    out_df["Svar1"] = pd.to_numeric(out_df["Svar1"], errors="coerce")
    out_df["Svar2"] = pd.to_numeric(out_df["Svar2"], errors="coerce")

    for col_name in ["B-HbA1c", "B-HbA1C", "B-HBA1C", "B-HBA1C, prosentvis"]:
        hba1c_percent_idx = (out_df["Tekst"] == col_name) & (out_df["Prøvesvar_ENHET"] == "%")
        out_df.loc[hba1c_percent_idx, "Svar1"] = (
            out_df.loc[hba1c_percent_idx, "Svar1"].astype(float) - 2.152
        ) / 0.09148
        out_df.loc[hba1c_percent_idx, "Svar2"] = (
            out_df.loc[hba1c_percent_idx, "Svar2"].astype(float) - 2.152
        ) / 0.09148
        out_df.loc[hba1c_percent_idx, "Prøvesvar_ENHET"] = "mmol/mol"
        out_df.loc[hba1c_percent_idx, "Tekst"] = "B-HbA1c"

    out_df = out_df[~out_df["Tekst"].str.startswith("U-")]
    out_df = out_df[~out_df["Tekst"].str.startswith("Plv-")]
    out_df = out_df[~out_df["Tekst"].str.startswith("BAL-")]

    out_df["result"] = out_df["Svar1"].combine_first(out_df["Svar2"])
    out_df = out_df.rename(columns={"Prøvesvar_ENHET": "unit", "ID": "id", "Tekst": "test_name"})

    out_df.rename(columns={"Dager_siden_BC_diagnose": "days_since_diagnosis"}, inplace=True)
    out_df["days_since_diagnosis"] = out_df["days_since_diagnosis"]
    return out_df[["id", "days_since_diagnosis", "test_name", "result", "unit"]]


def extract_clinical_events(source: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    df = source
    patients_df = endpoints
    clinical_df = (
        df.loc[
            df["ID"].isin(patients_df["id"]) & df["Datatype"].isin(["Diagnoser", "Prosedyrer"]),
            ["ID", "Dager_siden_BC_diagnose", "Datatype", "Kode", "Tekst"],
        ]
        .rename(
            columns={
                "ID": "id",
                "Dager_siden_BC_diagnose": "days_since_diagnosis",
                "Datatype": "event_type",
                "Kode": "code",
                "Tekst": "description",
            }
        )
        .copy()
    )
    clinical_df["event_type"] = clinical_df["event_type"].replace({"Diagnoser": "diagnosis", "Prosedyrer": "procedure"})
    clinical_df["days_since_diagnosis"] = pd.to_numeric(clinical_df["days_since_diagnosis"], errors="coerce").astype(
        "Int64"
    )
    clinical_df = clinical_df.dropna(subset=["id", "days_since_diagnosis", "event_type"])

    return clinical_df[["id", "days_since_diagnosis", "event_type", "code", "description"]]
