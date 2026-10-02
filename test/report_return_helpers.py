from pathlib import Path
import shutil
from typing import Literal

from src.data.report_return_lookup import return_directory


def archive_return(path: Path, kind: Literal["ages", "reference_percentages"]) -> Path:
    destination = return_directory(kind) / path.name
    shutil.copy2(path, destination)
    return destination
