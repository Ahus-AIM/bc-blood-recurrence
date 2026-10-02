from __future__ import annotations

import json
from pathlib import Path

from src.script.ebm_survival_config import update_run_metadata


def test_update_metadata_replaces_explanation_artifacts(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"train": {"row_count": 3}, "explanations": {"old": True}}),
        encoding="utf-8",
    )

    update_run_metadata(tmp_path, "explanations", {"term_hazard_ratio_plots": [{"path": "explanations/terms_hazard_ratio/a.png"}]})

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["train"] == {"row_count": 3}
    assert metadata["explanations"] == {
        "term_hazard_ratio_plots": [{"path": "explanations/terms_hazard_ratio/a.png"}]
    }
