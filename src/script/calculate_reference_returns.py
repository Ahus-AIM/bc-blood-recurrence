"""Run the original reference calculator on fixed local patient dates and exported measurements."""

from __future__ import annotations

import runpy
import shutil

from src.paths import BENCHMARK, PROCESSED, RAW


def main() -> None:
    directory = BENCHMARK / "All/6m/__all_blood_tests__/remote_reference"
    calculator = directory / "calculate_recurrence_reference_percentages.py"
    patients = RAW / "patients.csv"
    if not calculator.is_file():
        raise FileNotFoundError("Train or regenerate recurrence statistics before calculating reference returns.")
    if not patients.is_file():
        raise FileNotFoundError(f"Exact patient dates are required at {patients}; see docs/inputs.md.")
    namespace = runpy.run_path(str(calculator))
    namespace["REMOTE_SETTINGS"]["patients_csv"] = str(patients)
    result = namespace["run_calculation"](directory)
    destination = PROCESSED / "reference_percentages"
    destination.mkdir(parents=True, exist_ok=True)
    # One file per data hash allows different experiments to coexist.
    data_hash = namespace["EXPORT_MANIFEST"]["data_hash"]
    shutil.copy2(result, destination / f"{data_hash}.csv")


if __name__ == "__main__":
    main()
