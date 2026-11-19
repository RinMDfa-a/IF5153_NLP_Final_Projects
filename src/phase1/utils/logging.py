"""Logging helpers. Every script logs row counts in and out."""
from __future__ import annotations

import logging as _logging


def get_logger(name: str) -> _logging.Logger:
    logger = _logging.getLogger(name)
    if not logger.handlers:
        handler = _logging.StreamHandler()
        handler.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(_logging.INFO)
        logger.propagate = False
    return logger


def log_rows(logger: _logging.Logger, step: str, n_in: int, n_out: int) -> None:
    logger.info("%s: %d rows in, %d rows out (%d dropped)", step, n_in, n_out, n_in - n_out)
