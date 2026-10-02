from __future__ import annotations

from pathlib import Path

from src.script.survival_benchmark_artifacts import file_sha256, load_completed_results, write_completion_marker


def test_resume_requires_matching_run_and_model_input_fingerprints(tmp_path: Path) -> None:
    ebm_input = tmp_path / "model_input.csv"
    ebm_input.write_text("feature\n1\n", encoding="utf-8")
    input_fingerprint = file_sha256(ebm_input)
    results = [{"model_key": "A", "split": "test", "status": "complete", "harrell_c_index": 0.5}]

    write_completion_marker(tmp_path, "run-fingerprint", input_fingerprint, results)

    assert load_completed_results(tmp_path, "run-fingerprint", input_fingerprint) == results
    assert load_completed_results(tmp_path, "changed", input_fingerprint) is None
    ebm_input.write_text("feature\n2\n", encoding="utf-8")
    assert load_completed_results(tmp_path, "run-fingerprint", file_sha256(ebm_input)) is None
