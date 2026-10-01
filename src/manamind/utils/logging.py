"""Logging setup for ManaMind."""

import logging
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    fmt: str = _DEFAULT_FORMAT,
) -> logging.Logger:
    """Configure the root ``manamind`` logger.

    Args:
        level: Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        log_file: Optional path to also write logs to. Parent directories are
            created if needed.
        fmt: Format string for log records.

    Returns:
        The configured ``manamind`` logger.
    """
    logger = logging.getLogger("manamind")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers when called more than once.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
