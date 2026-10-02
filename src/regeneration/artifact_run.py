"""Record manuscript files written by completed tasks in the latest regeneration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from src.regeneration.context import RegenerationContext
from src.regeneration.paper_artifacts import PaperArtifact
from src.reporting import display_output_path
from src.script.survival_benchmark_artifacts import file_sha256

RECORD_FILENAME = "regeneration_artifacts.json"
RunStatus = Literal["running", "complete", "failed"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class GeneratedArtifact:
    """A completed task's output and its content at task completion.

    ``source`` is benchmark-relative or absolute; ``destination`` is bundle-relative.
    ``sha256`` prevents replaying a later overwrite. ``task`` identifies the producer;
    ``generated_utc`` records completion and ``source_modified_utc`` the file timestamp.
    """

    source: str
    destination: str
    sha256: str
    task: str
    generated_utc: str
    source_modified_utc: str


@dataclass
class ArtifactRun:
    """The latest producing run, retained independently of collection-only runs.

    ``benchmark_dir`` owns the record; ``horizon`` fixes its manuscript scope.
    ``run_id`` and ``started_utc`` identify the run. ``files`` holds the latest
    successful output per destination. Status, completed tasks and failed task
    track progress, including when automatic collection is disabled.
    """

    benchmark_dir: Path
    horizon: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_utc: str = field(default_factory=utc_now)
    status: RunStatus = "running"
    completed_tasks: list[str] = field(default_factory=list)
    failed_task: str | None = None
    files: list[GeneratedArtifact] = field(default_factory=list)

    def save(self) -> None:
        payload = asdict(self)
        payload.update(format_version=1, benchmark_dir=str(self.benchmark_dir))
        write_json(self.benchmark_dir / RECORD_FILENAME, payload)

    def record(self, task: str, files: tuple[GeneratedArtifact, ...]) -> None:
        destinations = {item.destination: item for item in self.files}
        destinations.update((item.destination, item) for item in files)
        self.files = list(destinations.values())
        self.completed_tasks.append(task)
        self.save()


def load_artifact_run(context: RegenerationContext) -> ArtifactRun:
    path = context.output_dir / RECORD_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No recorded regeneration outputs at {path}. Run the producing tasks with output tracking first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or payload["status"] not in ("running", "complete", "failed"):
        raise ValueError(f"Unsupported regeneration artifact record: {path}")
    if payload["horizon"] != context.settings.paper_bundle.horizon:
        raise ValueError("The recorded regeneration horizon differs from paper_bundle.horizon.")
    run = ArtifactRun(
        benchmark_dir=context.output_dir,
        horizon=payload["horizon"],
        run_id=payload["run_id"],
        started_utc=payload["started_utc"],
        status=cast(RunStatus, payload["status"]),
        completed_tasks=payload["completed_tasks"],
        failed_task=payload["failed_task"],
        files=[GeneratedArtifact(**item) for item in payload["files"]],
    )
    if any(item.task not in run.completed_tasks for item in run.files):
        raise ValueError("Regeneration artifact record contains an unfinished task's output.")
    return run


def generated_artifacts(
    root: Path,
    task: str,
    artifacts: tuple[PaperArtifact, ...],
    written: tuple[Path, ...],
) -> tuple[GeneratedArtifact, ...]:
    """Keep only allowed manuscript outputs explicitly returned by the producer."""
    paths = {path.resolve() for path in written}
    records = []
    for artifact in artifacts:
        if artifact.unavailable is not None or artifact.source.resolve() not in paths:
            continue
        records.append(GeneratedArtifact(
            source=display_output_path(artifact.source, root),
            destination=str(artifact.destination),
            sha256=file_sha256(artifact.source),
            task=task,
            generated_utc=utc_now(),
            source_modified_utc=datetime.fromtimestamp(artifact.source.stat().st_mtime, timezone.utc).isoformat(),
        ))
    return tuple(records)
