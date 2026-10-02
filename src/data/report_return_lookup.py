"""Discover and validate external report numbers in shared CSV directories."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

from src.config import load_yml, resolve_path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/report_returns.yml"
LOGGER = logging.getLogger(__name__)
ReturnValue = TypeVar("ReturnValue")


@dataclass(frozen=True)
class ReportReturn(Generic[ReturnValue]):
    """Validated ``value`` and all equivalent matching CSV ``paths``, in filename order."""

    value: ReturnValue
    paths: tuple[Path, ...]


def return_directory(kind: Literal["ages", "reference_percentages"]) -> Path:
    settings = load_yml(CONFIG_PATH)["directories"]
    return resolve_path(str(settings[kind]), CONFIG_PATH.parent)


def csv_matches_hash(path: Path, column: str, expected_hash: str) -> bool:
    """Skip unidentifiable files; malformed rows in an identifiable match are errors."""
    matches = False
    has_hash = False
    malformed = False
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            header = next(reader, [])
            positions = [index for index, name in enumerate(header) if name == column]
            if not positions:
                LOGGER.warning("Skipping report return %s: no %s column.", path, column)
                return False
            malformed = len(set(header)) != len(header)
            for row in reader:
                if not row:
                    continue
                hashes = [row[index] for index in positions if index < len(row)]
                matches |= expected_hash in hashes
                has_hash |= any(value.strip() for value in hashes)
                malformed |= len(row) != len(header)
    except (OSError, UnicodeError, csv.Error) as error:
        if matches:
            raise ValueError(f"Cannot read matching report return {path}: {error}") from error
        LOGGER.warning("Skipping unreadable report return %s: %s", path, error)
        return False
    if malformed and matches:
        raise ValueError(f"Malformed CSV in matching report return {path}.")
    if malformed or not has_hash:
        LOGGER.warning("Skipping report return %s: malformed CSV or no usable %s.", path, column)
        return False
    return matches


def find_report_return(
    directory: Path,
    hash_column: str,
    expected_hash: str,
    *,
    load: Callable[[Path], ReturnValue],
    equivalent: Callable[[ReturnValue, ReturnValue], bool],
) -> ReportReturn[ReturnValue] | None:
    """Validate every matching file, accepting duplicates only when their values agree."""
    paths = tuple(path for path in sorted(directory.glob("*.csv"))
                  if path.is_file() and csv_matches_hash(path, hash_column, expected_hash))
    if not paths:
        LOGGER.warning("No matching report return in %s for %s=%s; values remain unavailable.",
                       directory, hash_column, expected_hash)
        return None
    values = [load(path) for path in paths]
    for path, value in zip(paths[1:], values[1:], strict=True):
        if not equivalent(values[0], value):
            raise ValueError(f"Conflicting report returns for {expected_hash}: {paths[0]} and {path}.")
    LOGGER.info("Using report return for %s=%s from %s", hash_column, expected_hash, ", ".join(map(str, paths)))
    return ReportReturn(values[0], paths)
