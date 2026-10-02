"""Run the four study notebook exports, in order, using fixed repository paths."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import load_yml
from src.extraction.blood import extract_blood_tests, extract_clinical_events
from src.extraction.diagnoses import extract_diagnosis_days
from src.extraction.endpoints import extract_endpoints
from src.extraction.splits import extract_splits, parse_recurrence_days
from src.paths import CONFIG, OUTPUT, PROCESSED, RAW
from src.script.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def dated_source(source: pd.DataFrame) -> pd.DataFrame:
    """Apply the original timedelta parsing before selection, without changing source order."""
    frame = source.copy()
    column = "Dager_siden_BC_diagnose"
    frame[column] = pd.to_timedelta(frame[column]).dt.days
    frame = frame.dropna(subset=[column])
    frame[column] = frame[column].astype("Int64")
    return frame


def main() -> None:
    for name in ("dips.csv", "to_keep.csv"):
        if not (RAW / name).is_file():
            raise FileNotFoundError(f"Place the source export at {RAW / name}; see docs/inputs.md.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    configure_logging(OUTPUT / "extraction.log")
    source = pd.read_csv(RAW / "dips.csv")
    selection = pd.read_csv(RAW / "to_keep.csv")
    selection["to_keep"] = selection["to_keep"].astype(int).astype(bool)
    selected_ids = selection.loc[selection["to_keep"], "ID"]
    endpoints = extract_endpoints(source, selected_ids)
    endpoints.to_csv(PROCESSED / "bc_endpoints.csv", index=False)

    # The notebooks reload this CSV; preserve that dtype/serialization boundary.
    endpoints = pd.read_csv(PROCESSED / "bc_endpoints.csv")
    dated = dated_source(source)
    selected = dated[dated["ID"].isin(selected_ids)].reset_index(drop=True)
    blood = extract_blood_tests(selected, endpoints)
    blood.to_csv(PROCESSED / "blood_test.csv", index=False)
    extract_clinical_events(selected, endpoints).to_csv(PROCESSED / "diagnosis_procedure.csv", index=False)

    endpoints = pd.read_csv(
        PROCESSED / "bc_endpoints.csv", converters={"diagnosis_to_ddfs_in_days": parse_recurrence_days}
    )
    blood = pd.read_csv(PROCESSED / "blood_test.csv")
    settings = load_yml(CONFIG / "ebm_survival.yml")["dataset"]
    extract_splits(endpoints, blood, settings).to_csv(PROCESSED / "bc_split.csv", index=False)
    extract_diagnosis_days(dated).to_csv(PROCESSED / "bc_diagnosis_days.csv", index=False)
    LOGGER.info("Wrote five notebook-equivalent CSVs to %s", PROCESSED)
    if not (PROCESSED / "bc_diagnosis_patient_age.csv").is_file():
        LOGGER.warning("Model ages must be supplied separately; the original age-calculation code is pending.")


if __name__ == "__main__":
    main()
