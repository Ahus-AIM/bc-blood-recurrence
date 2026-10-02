"""Compare generated artifacts with the frozen manuscript and optional local reference exports."""

from __future__ import annotations

import json
from pathlib import Path

from src.paths import BENCHMARK, CONFIG, OUTPUT, PAPER, PROCESSED, REFERENCE
from src.reproduction import artifact_hash, compare_files

EXTRACTION_FILES = (
    "bc_endpoints.csv", "blood_test.csv", "diagnosis_procedure.csv", "bc_split.csv", "bc_diagnosis_days.csv"
)


def manuscript_comparisons() -> dict[str, dict[str, object]]:
    snapshot = json.loads((CONFIG / "manuscript_reference.json").read_text())
    comparisons = {}
    for name, expected in snapshot["artifacts"].items():
        path = PAPER / name
        if not path.is_file():
            comparisons[name] = {"status": "missing_output"}
            continue
        actual = artifact_hash(path)
        comparisons[name] = {
            "status": "match" if actual == expected["comparison_sha256"] else "different",
            "expected_sha256": expected["comparison_sha256"],
            "actual_sha256": actual,
        }
    return comparisons


def reference_comparisons(reference: Path, actual: Path, names: list[Path]) -> dict[str, dict[str, object]]:
    return {str(name): compare_files(reference / name, actual / name) for name in names}


def main() -> None:
    manuscript = manuscript_comparisons()
    processed = {
        name: compare_files(REFERENCE / "processed" / name, PROCESSED / name)
        if (REFERENCE / "processed" / name).is_file() else {"status": "missing_reference"}
        for name in EXTRACTION_FILES
    }
    benchmark_reference = REFERENCE / "benchmark"
    names = [path.relative_to(benchmark_reference) for path in sorted(benchmark_reference.rglob("*.csv"))]
    benchmark = reference_comparisons(benchmark_reference, BENCHMARK, names)
    matches = all(item["status"] == "match" for item in manuscript.values())
    report = {
        "synthetic": (OUTPUT / "SYNTHETIC.json").is_file(),
        "manuscript_matches": matches,
        "manuscript": manuscript,
        "extraction": processed,
        "benchmark_csvs": benchmark,
        "benchmark_reference_supplied": bool(names),
        "scope": "SVG rendering structure and LaTeX tabular bodies; optional extraction and benchmark CSV bytes. "
                 "This does not establish clinical validity or compare fitted model internals.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Manuscript: {sum(row['status'] == 'match' for row in manuscript.values())}/{len(manuscript)} match.")
    print(f"Details: {destination}")
    failures = any(row["status"] != "match" for row in benchmark.values())
    failures |= any(row["status"] not in ("match", "missing_reference") for row in processed.values())
    if not matches or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
