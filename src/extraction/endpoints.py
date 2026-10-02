"""Endpoint export; notebook 01 operations and ordering are preserved."""

from __future__ import annotations

import pandas as pd

from src.data.endpoint_extraction import first_recurrence_records, log_endpoint_export, prepare_endpoint_source


def extract_endpoints(source: pd.DataFrame, selected_ids: pd.Series) -> pd.DataFrame:
    """Retain each first recurrence row and cap observation/recurrence offsets at death."""
    dips_df = prepare_endpoint_source(source, selected_ids)
    death_df = dips_df[dips_df["Kode"] == "Død"]

    death_df = death_df[["ID", "Dager_siden_BC_diagnose"]]

    death_df.rename(columns={"Dager_siden_BC_diagnose": "days_from_diagnosis_to_death"}, inplace=True)

    ddfs_events = capped_recurrence_records(dips_df, death_df)

    ddfs_lists_df = (
        ddfs_events.sort_values(["ID", "Dager_siden_BC_diagnose", "Kode"])
        .groupby("ID")
        .agg(
            diagnosis_to_ddfs_in_days=("Dager_siden_BC_diagnose", lambda values: [int(value) for value in values]),
            ddfs_code=("Kode", lambda values: [str(value) for value in values]),
            ddfs_description=("Tekst", lambda values: [str(value) for value in values]),
        )
        .reset_index()
    )

    last_entry_df = (
        dips_df.groupby("ID")[["Dager_siden_BC_diagnose", "BC_diagnose_år"]]
        .max()
        .reset_index()
        .rename(
            columns={
                "Dager_siden_BC_diagnose": "days_from_diagnosis_to_last_entry",
                "BC_diagnose_år": "BC_diagnosis_year",
            }
        )
    )

    endpoint_df = dips_df[["ID", "BC_diagnose_år"]].drop_duplicates().merge(ddfs_lists_df, how="left", on="ID")

    endpoint_df = endpoint_df.merge(death_df, how="left", on="ID")

    endpoint_df = endpoint_df.merge(last_entry_df[["ID", "days_from_diagnosis_to_last_entry"]], how="left", on="ID")

    endpoint_df["days_from_diagnosis_to_last_entry"] = endpoint_df["days_from_diagnosis_to_last_entry"]

    endpoint_df["days_from_diagnosis_to_death"] = endpoint_df["days_from_diagnosis_to_death"]

    endpoint_df["days_from_diagnosis_to_last_entry"] = endpoint_df[
        ["days_from_diagnosis_to_last_entry", "days_from_diagnosis_to_death"]
    ].min(axis=1, skipna=True)

    endpoint_df.rename(columns={"ID": "id"}, inplace=True)

    endpoint_df.rename(columns={"BC_diagnose_år": "bc_diagnosis_year"}, inplace=True)
    log_endpoint_export(dips_df, endpoint_df)
    return endpoint_df


def capped_recurrence_records(source: pd.DataFrame, deaths: pd.DataFrame) -> pd.DataFrame:
    """Keep the original whole first recurrence record, moving post-death offsets to death day."""
    events = first_recurrence_records(source).merge(deaths, how="left")
    events["Dager_siden_BC_diagnose"] = events[
        ["Dager_siden_BC_diagnose", "days_from_diagnosis_to_death"]
    ].min(axis=1)
    return events.drop(columns=["days_from_diagnosis_to_death"])
