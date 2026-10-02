from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOG_FILENAME = "run.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_MARKER = "_pathology_blood_ror_logging_handler"
_BUFFERED_RECORD_LIMIT = 1_000


class BufferedLogHandler(logging.Handler):
    """Keeps early log records until the configured run output directory is known."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if len(self.records) > _BUFFERED_RECORD_LIMIT:
            self.records.pop(0)


def configure_logging(log_path: Path | None = None) -> Path | None:
    """Configure console logging and optionally mirror log records to a file."""
    root_logger = logging.getLogger()
    buffered_records = buffered_log_records(root_logger)
    remove_configured_handlers(root_logger)
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT)
    add_configured_handler(root_logger, logging.StreamHandler(), formatter)
    if log_path is None:
        add_configured_handler(root_logger, BufferedLogHandler(), formatter)
        resolved_log_path = None
    else:
        resolved_log_path = log_path.resolve()
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_path, mode="w", encoding="utf-8")
        add_configured_handler(root_logger, file_handler, formatter)
        replay_buffered_records(file_handler, buffered_records)

    logging.getLogger("interpret").setLevel(logging.WARNING)
    if resolved_log_path is not None:
        logging.getLogger(__name__).info("storing logs in %s", resolved_log_path)
    return resolved_log_path

def add_configured_handler(
    logger: logging.Logger,
    handler: logging.Handler,
    formatter: logging.Formatter,
) -> None:
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    setattr(handler, _HANDLER_MARKER, True)
    logger.addHandler(handler)


def remove_configured_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if bool(getattr(handler, _HANDLER_MARKER, False)):
            logger.removeHandler(handler)
            handler.close()


def buffered_log_records(logger: logging.Logger) -> list[logging.LogRecord]:
    records: list[logging.LogRecord] = []
    for handler in logger.handlers:
        if isinstance(handler, BufferedLogHandler):
            records.extend(handler.records)
    return records


def replay_buffered_records(
    file_handler: logging.FileHandler,
    records: list[logging.LogRecord],
) -> None:
    for record in records:
        file_handler.handle(record)
    file_handler.flush()
