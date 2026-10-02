"""Generate invented raw records for exercising the pipeline; never overwrite existing inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_yml
from src.paths import CONFIG, OUTPUT, PROCESSED, RAW

COLUMNS = (
    "ID",
    "Datatype",
    "Kode",
    "Tekst",
    "Dager_siden_BC_diagnose",
    "BC_diagnose_år",
    "Svar1",
    "Svar2",
    "Prøvesvar_ENHET",
)
RECURRENCE_CODES = ("C795", "C793", "C787", "C780")


def blood_panel() -> pd.DataFrame:
    ranges = pd.read_csv(CONFIG / "ahus_blood_test_normal_ranges.csv")
    return ranges.loc[ranges["sex"].eq("female")].drop_duplicates(["test_name", "unit"]).reset_index(drop=True)


def synthetic_frames(patient_count: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Invent all records from a seed; no clinical source or manuscript measurements are read."""
    if patient_count < 72 or patient_count % 3:
        raise ValueError("Use at least 72 synthetic patients, with a count divisible by three.")
    random = np.random.default_rng(seed)
    panel = blood_panel()
    records: list[list[object]] = []
    ages: list[dict[str, object]] = []
    dates: list[dict[str, object]] = []
    for index in range(patient_count):
        patient = f"SYN{index:04d}"
        year = (2010, 2019, 2023)[index // (patient_count // 3)]
        age = int(random.integers(35, 81))
        diagnosis_date = pd.Timestamp(year, 1, 1)
        dates.append(
            {
                "id": patient,
                "birth date": f"{year - age}-01-01",
                "day of first bc diagnosis": str(diagnosis_date.date()),
            }
        )
        ages.append({"id": patient, "age": age // 5 * 5})
        event_day = 640 + index % 17
        outcome = index % 6

        def append(
            day: int, kind: str, code: str, text: str, first: object = None, second: object = None, unit: object = None
        ) -> None:
            records.append([patient, kind, code, text, f"{day} days", year, first, second, unit])

        append(0, "Diagnoser", "C50", "Synthetic initial diagnosis")
        if outcome < len(RECURRENCE_CODES):
            append(event_day, "Diagnoser", RECURRENCE_CODES[outcome], "Synthetic distant recurrence")
            append(event_day + 200 + index % 30, "Død", "", "Synthetic death")
        elif outcome == 4:
            append(event_day, "Død", "", "Synthetic competing death")
        append(event_day + 400, "Prosedyrer", "TEST", "Synthetic last observation")
        days = sorted(
            {-20, 60, 200, event_day - 300, event_day - 120, event_day - 60, event_day - 20, event_day - 3, event_day}
        )
        for day in days:
            for test_index, test in panel.iterrows():
                if day > 0 and (index + day + test_index) % 29 == 0:
                    continue
                lower = float(test["lower"]) if pd.notna(test["lower"]) else 0.0
                upper = float(test["upper"]) if pd.notna(test["upper"]) else max(lower * 1.3, 10.0)
                center = (lower + upper) / 2
                progression = max(0, day - event_day + 180) / 180
                shift = progression * (0.2 if outcome < 4 else -0.03)
                value = max(0.001, center * (1 + shift + random.normal(0, 0.1)))
                result = f"{value:.6f}"
                # Exercise decimal separators and preference for Svar1/Svar2.
                if (index + test_index) % 7 == 0:
                    result = result.replace(".", ",")
                if (index + test_index) % 11 == 0:
                    append(day, "Blodprøver", "LAB", str(test["test_name"]), None, result, test["unit"])
                else:
                    append(day, "Blodprøver", "LAB", str(test["test_name"]), result, None, test["unit"])
    # Include a patient excluded by diagnosis year so every age-summary fold exists.
    records.append(["SYN-EXCLUDED", "Diagnoser", "C50", "Synthetic exclusion", "0 days", 2000, None, None, None])
    records.append(["SYN-EXCLUDED", "Prosedyrer", "TEST", "Synthetic follow-up", "900 days", 2000, None, None, None])
    ages.append({"id": "SYN-EXCLUDED", "age": 50})
    dates.append({"id": "SYN-EXCLUDED", "birth date": "1950-01-01", "day of first bc diagnosis": "2000-01-01"})
    return pd.DataFrame(records, columns=COLUMNS), pd.DataFrame(ages), pd.DataFrame(dates)


def require_empty(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Synthetic generation requires an empty directory: {directory}")


def main() -> None:
    for directory in (RAW, PROCESSED, OUTPUT):
        require_empty(directory)
    settings = load_yml(CONFIG / "synthetic.yml")["synthetic"]
    raw, ages, dates = synthetic_frames(int(settings["patients"]), int(settings["seed"]))
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW / "dips.csv", index=False)
    pd.DataFrame({"ID": ages["id"], "to_keep": 1}).to_csv(RAW / "to_keep.csv", index=False)
    dates.to_csv(RAW / "patients.csv", index=False)
    ages.to_csv(PROCESSED / "bc_diagnosis_patient_age.csv", index=False)
    (RAW / "SYNTHETIC.json").write_text(json.dumps({"synthetic": True, **settings}, indent=2) + "\n")
    print(f"Wrote invented demonstration data to {RAW}. These data cannot reproduce the manuscript.")


if __name__ == "__main__":
    main()
