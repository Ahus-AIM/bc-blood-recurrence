from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

COMPLETION_FILENAME = "benchmark_completion.json"
LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_sha256(value: Mapping[str, object]) -> str:
    """Hash a JSON-compatible mapping with stable key ordering."""
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_completion_marker(
    output_dir: Path,
    run_fingerprint: str,
    model_input_fingerprint: str,
    results: Sequence[Mapping[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_fingerprint": run_fingerprint,
        "model_input_fingerprint": model_input_fingerprint,
        "results": [dict(result) for result in results],
    }
    (output_dir / COMPLETION_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_completed_results(
    output_dir: Path,
    run_fingerprint: str,
    model_input_fingerprint: str,
) -> list[dict[str, object]] | None:
    completion_path = output_dir / COMPLETION_FILENAME
    if not completion_path.is_file():
        return None
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Completion marker must contain a mapping: {completion_path}")
    if payload.get("run_fingerprint") != run_fingerprint:
        return None
    if payload.get("model_input_fingerprint") != model_input_fingerprint:
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not all(isinstance(result, Mapping) for result in results):
        raise TypeError(f"Completion marker results must be a list of mappings: {completion_path}")
    return [dict(result) for result in results]


def remove_completion_marker(output_dir: Path) -> None:
    """Invalidate resumable state before or after an unsuccessful run attempt."""
    try:
        (output_dir / COMPLETION_FILENAME).unlink(missing_ok=True)
    except OSError:
        LOGGER.exception("could not remove benchmark completion marker: %s", output_dir)
