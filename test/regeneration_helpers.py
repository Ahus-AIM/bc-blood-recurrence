from pathlib import Path
from collections.abc import Mapping

import yaml


def regeneration_config(
    directory: Path,
    benchmark: Mapping[str, object],
    tasks: list[str] | str,
    *,
    output_dir: Path | None = None,
    **settings: object,
) -> Path:
    benchmark_path = directory / "regeneration_benchmark.yml"
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "output_dir": str(directory if output_dir is None else output_dir),
                    **benchmark,
                }
            }
        )
    )
    path = directory / "regenerate.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": {"benchmark_config_path": str(benchmark_path)},
                "regenerate": {"tasks": tasks, **settings},
            }
        )
    )
    return path
