from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from src.paths import apply_fixed_paths

__all__ = ["load_yml", "resolve_mapping_path", "resolve_path"]


def load_yml(config_path: Path) -> dict[str, dict[str, object]]:
    """Load the section/key YAML shape used by project configs."""
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Expected a mapping at the top level of {config_path}.")
    config: dict[str, dict[str, object]] = {}
    for section, values in loaded.items():
        if not isinstance(section, str):
            raise ValueError(f"Expected string section names in {config_path}.")
        if not isinstance(values, Mapping):
            raise ValueError(f"Expected section `{section}` in {config_path} to be a mapping.")
        config[section] = dict(values)
    apply_fixed_paths(config_path, config)
    return config


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_mapping_path(
    config: Mapping[str, object],
    key: str,
    default_name: str,
    base_dir: Path,
) -> Path:
    """Resolve one optional mapping path relative to ``base_dir``."""
    return resolve_path(str(config.get(key, default_name)), base_dir)
