from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def shared_report_returns(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Keep report discovery independent of clinical CSVs stored in the workspace."""
    directory_root = tmp_path_factory.mktemp("report_returns")
    directories = {kind: directory_root / kind for kind in ("ages", "reference_percentages")}
    for directory in directories.values():
        directory.mkdir()
    config = directory_root / "report_returns.yml"
    config.write_text(yaml.safe_dump({"directories": {key: str(path) for key, path in directories.items()}}))
    monkeypatch.setattr("src.data.report_return_lookup.CONFIG_PATH", config)
    return directories
