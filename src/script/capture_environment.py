"""Record package versions and the native InterpretML binary used by this environment."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import sys

from src.paths import OUTPUT, ROOT
from src.reproduction import sha256


def main() -> None:
    distribution = metadata.distribution("interpret-core")
    native = {
        str(path): sha256(distribution.locate_file(path).read_bytes())
        for path in distribution.files
        if str(path).endswith((".so", ".dll", ".dylib"))
    }
    direct_url = distribution.read_text("direct_url.json")
    provenance = json.loads((ROOT / "docs/source_provenance.json").read_text())
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "interpret_native_sha256": native,
        "interpret_direct_url": None if direct_url is None else json.loads(direct_url),
        "required_clinical_interpret_commit": provenance["clinical_interpret_commit"],
        "lock_sha256": sha256((ROOT / "uv.lock").read_bytes()),
        "thread_settings": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED"
        )},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "environment.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Environment recorded at {destination}; package version alone does not identify the clinical native build.")


if __name__ == "__main__":
    main()
