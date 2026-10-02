from __future__ import annotations

import logging
import re

import pandas as pd

from src.data.icd_codes import compact_icd_code, matches_icd_code

LOGGER = logging.getLogger(__name__)
DIAGNOSIS_DAY_COLUMN = "Dager_siden_BC_diagnose"


def prepare_endpoint_source(source: pd.DataFrame, ids_to_keep: pd.Series) -> pd.DataFrame:
    """Apply the notebook's existing date/ID filters and account for missing patients."""
    days = pd.to_timedelta(source[DIAGNOSIS_DAY_COLUMN]).dt.days
    selected = source["ID"].isin(ids_to_keep)
    dated = days.notna()
    log_source_selection(source, ids_to_keep, selected, dated)
    retained = source.loc[selected & dated].copy()
    retained[DIAGNOSIS_DAY_COLUMN] = days.loc[selected & dated].astype("Int64")
    retained.loc[retained["Datatype"] == "Død", ["Kode", "Tekst"]] = "Død"
    return retained.reset_index(drop=True)


def log_source_selection(
    source: pd.DataFrame,
    ids_to_keep: pd.Series,
    selected: pd.Series,
    dated: pd.Series,
) -> None:
    """Partition selected unique IDs into absent, undated and retained patients."""
    selected_counts = ids_to_keep.value_counts()
    selected_ids = selected_counts.index
    raw_counts = source["ID"].value_counts()
    dated_counts = source.loc[dated, "ID"].value_counts()
    absent = selected_ids[~selected_ids.isin(raw_counts.index)]
    undated = selected_ids[selected_ids.isin(raw_counts.index) & ~selected_ids.isin(dated_counts.index)]
    retained_count = int(selected_ids.isin(dated_counts.index).sum())
    LOGGER.info(
        "to_keep selection: %d rows; %d unique patients; %d repeated IDs; %d missing-ID rows",
        len(ids_to_keep), len(selected_ids), int(ids_to_keep.dropna().duplicated().sum()), int(ids_to_keep.isna().sum()),
    )
    LOGGER.info("duplicate selected patient IDs: %s", sorted(selected_counts[selected_counts > 1].index, key=str))
    LOGGER.info(
        "raw DIPS: %d records; %d unique patients; %d missing-ID rows; ID dtype=%s; to_keep ID dtype=%s",
        len(source), len(raw_counts), int(source["ID"].isna().sum()), source["ID"].dtype, ids_to_keep.dtype,
    )
    LOGGER.info("selected patients absent from raw DIPS: %d", len(absent))
    LOGGER.info("selected patients with no dated DIPS records: %d", len(undated))
    for reason, patient_ids in (("absent from raw DIPS", absent), ("no dated DIPS records", undated)):
        for patient_id in sorted(patient_ids, key=str):
            LOGGER.info(
                "reason=%s; patient_id=%r; raw_records=%d; dated_records=%d",
                reason, patient_id, int(raw_counts.get(patient_id, 0)), int(dated_counts.get(patient_id, 0)),
            )
    LOGGER.info(
        "selected DIPS records: %d; missing-date records removed: %d; retained records: %d",
        int(selected.sum()), int((selected & ~dated).sum()), int((selected & dated).sum()),
    )
    LOGGER.info(
        "source reconciliation: %d selected = %d absent + %d without dates + %d retained patients",
        len(selected_ids), len(absent), len(undated), retained_count,
    )


def log_endpoint_export(source: pd.DataFrame, endpoints: pd.DataFrame) -> None:
    """Compare the final export with the dated, selected DIPS population without filtering it."""
    expected_ids = pd.Index(source["ID"].dropna().unique())
    exported_counts = endpoints["id"].value_counts()
    exported_ids = exported_counts.index
    lost = expected_ids[~expected_ids.isin(exported_ids)]
    unexpected = exported_ids[~exported_ids.isin(expected_ids)]
    LOGGER.info(
        "endpoint export: %d rows; %d unique patients; %d repeated IDs; %d missing-ID rows",
        len(endpoints), len(exported_ids), int(endpoints["id"].dropna().duplicated().sum()),
        int(endpoints["id"].isna().sum()),
    )
    LOGGER.info("duplicate endpoint patient IDs: %s", sorted(exported_counts[exported_counts > 1].index, key=str))
    LOGGER.info(
        "patients lost between dated DIPS and endpoint export: %d; IDs=%s", len(lost), sorted(lost, key=str),
    )
    LOGGER.info("unexpected endpoint patients: %d; IDs=%s", len(unexpected), sorted(unexpected, key=str))


def first_recurrence_records(source: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest whole source row per patient and compact C78/C79 code."""
    code_regex = re.compile(r"C7[89]\d*")
    codes = source["Kode"].fillna("").astype(str)
    recurrence = (source["Datatype"] == "Diagnoser") & codes.map(lambda code: matches_icd_code(code, code_regex))
    events = source.loc[recurrence].copy()
    events["_recurrence_code"] = codes.loc[recurrence].map(compact_icd_code)
    return (
        events.sort_values(["ID", "Dager_siden_BC_diagnose", "_recurrence_code", "Kode"], kind="stable")
        .drop_duplicates(["ID", "_recurrence_code"])
        .drop(columns="_recurrence_code")
        .reset_index(drop=True)
    )
