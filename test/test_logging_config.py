from __future__ import annotations

import logging
from pathlib import Path

from src.script.logging_config import configure_logging, remove_configured_handlers


def test_configure_logging_writes_log_records_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    try:
        logger = logging.getLogger("test.logging_config")
        configure_logging()
        logger.info("stored early log message")
        configure_logging(log_path)
        logger.info("stored later log message")
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_text = log_path.read_text(encoding="utf-8")
        assert "stored early log message" in log_text
        assert "stored later log message" in log_text
    finally:
        remove_configured_handlers(logging.getLogger())
