"""Update a cumulative manuscript bundle using recorded successful outputs only."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from src.regeneration.artifact_run import ArtifactRun, GeneratedArtifact, utc_now, write_json
from src.regeneration.context import RegenerationContext
from src.regeneration.paper_artifacts import bundle_destination, paper_artifacts
from src.reporting import report_output_path
from src.script.survival_benchmark_artifacts import file_sha256

LOGGER = logging.getLogger(__name__)


def collect_paper_bundle(
    context: RegenerationContext, run: ArtifactRun, files: tuple[GeneratedArtifact, ...],
) -> None:
    """Copy validated successful outputs, retaining every older destination."""
    artifacts = paper_artifacts(context)
    destination = bundle_destination(context, artifacts)
    allowed = {(item.source.resolve(), str(item.destination)): item.unavailable for item in artifacts}
    destination.mkdir(parents=True, exist_ok=True)
    updates: dict[str, dict[str, object]] = {}
    skipped: dict[str, str] = {}
    with TemporaryDirectory(prefix=".paper-bundle-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for item in files:
            source = report_output_path(context.output_dir, item.source)
            key = (source.resolve(), item.destination)
            if key not in allowed:
                raise ValueError(f"Recorded output is not a configured manuscript artifact: {item.source}")
            reason = allowed[key]
            if reason is None and not source.is_file():
                reason = "Recorded source is missing"
            if reason is None:
                target = staging / item.destination
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if file_sha256(target) != item.sha256:
                    reason = "Source changed since the successful task completed"
            if reason is not None:
                skipped[item.destination] = reason
                continue
            updates[item.destination] = {
                **asdict(item), "run_id": run.run_id, "horizon": run.horizon, "last_refreshed_utc": utc_now(),
            }
        for name in updates:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).replace(target)
    update_bundle_manifest(context, run, updates, skipped)
    for name, reason in skipped.items():
        LOGGER.warning("Paper bundle kept its previous copy of %s, if present: %s", name, reason)
    LOGGER.info("Collected %d generated manuscript files in %s", len(updates), destination)


def update_bundle_manifest(
    context: RegenerationContext,
    run: ArtifactRun,
    updates: dict[str, dict[str, object]] | None = None,
    skipped: dict[str, str] | None = None,
) -> None:
    """Describe current and retained files without deleting or recopying them."""
    artifacts = paper_artifacts(context)
    destination = bundle_destination(context, artifacts)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "manifest.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    records = {item["destination"]: item for item in previous.get("files", [])}
    if updates is not None:
        records.update(updates)
    omissions = dict(previous.get("uncollected", {})) if previous.get("run_id") == run.run_id else {}
    if skipped is not None:
        omissions.update(skipped)
    if updates is not None:
        for name in updates:
            omissions.pop(name, None)
    expected = sorted({str(item.destination) for item in artifacts})
    existing: list[dict[str, object]] = []
    missing = []
    for name in expected:
        target = destination / name
        if not target.is_file():
            missing.append(name)
            continue
        record = {"destination": name, "last_refreshed_utc": None, **records.get(name, {})}
        digest = file_sha256(target)
        if record.get("sha256") != digest:
            record = {"destination": name, "last_refreshed_utc": None, "sha256": digest}
        record["status"] = "refreshed" if record.get("run_id") == run.run_id else "retained"
        existing.append(record)
    write_json(manifest, {
        "updated_utc": utc_now(),
        "benchmark_dir": str(context.output_dir),
        "horizon": run.horizon,
        "run_id": run.run_id,
        "regeneration_status": run.status,
        "failed_task": run.failed_task,
        "status": "complete" if run.status == "complete" and not missing and not omissions else "incomplete",
        "refreshed": [item["destination"] for item in existing if item["status"] == "refreshed"],
        "retained": [item["destination"] for item in existing if item["status"] == "retained"],
        "missing": missing,
        "uncollected": omissions,
        "files": existing,
    })
