"""Logging helpers for MILGRAU command-line pipelines."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(module_name: str, log_dir: str = "logs") -> logging.Logger:
    """Create a standardized UTF-8 logger for a MILGRAU module."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{module_name}_run_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
