from pathlib import Path
from types import SimpleNamespace

import pytest

from src.script import ebm_training_artifacts


@pytest.mark.parametrize(
    ("imputation_enabled", "missing_bins_requested", "expected"),
    (
        (True, True, False),
        (True, False, False),
        (False, True, True),
        (False, False, False),
    ),
)
def test_configured_explanations_resolve_missing_bins_against_imputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    imputation_enabled: bool,
    missing_bins_requested: bool,
    expected: bool,
) -> None:
    captured: dict[str, object] = {}

    def capture_explanation_settings(*_args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        ebm_training_artifacts,
        "write_ebm_explanations",
        capture_explanation_settings,
    )

    ebm_training_artifacts.write_configured_ebm_explanations(
        model=object(),  # type: ignore[arg-type]
        output_dir=tmp_path,
        dataset_config=SimpleNamespace(  # type: ignore[arg-type]
            recurrence_endpoint_name="distant_recurrence"
        ),
        output_config={"explanation_include_missing_bins": missing_bins_requested},
        config_dir=tmp_path,
        imputation_enabled=imputation_enabled,
    )

    assert captured["include_missing_bins"] is expected
