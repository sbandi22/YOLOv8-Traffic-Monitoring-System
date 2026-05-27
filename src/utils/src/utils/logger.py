"""Centralised logging configuration using Loguru.

Provides a single ``get_logger`` entry point used across the project so logs
are consistent (file rotation, JSON structured output, console formatting)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_CONFIGURED = False


def configure_logger(
    log_file: str = "logs/traffic.log",
    level: str = "INFO",
    rotation: str = "20 MB",
    retention: str = "14 days",
    json_logs: bool = True,
) -> None:
    """Configure the global Loguru logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level=level,
        rotation=rotation,
        retention=retention,
        enqueue=True,
        serialize=json_logs,
        backtrace=True,
    )

    _CONFIGURED = True
    logger.debug("Logger configured (level={}, json={})", level, json_logs)


def get_logger(name: Optional[str] = None):
    """Return the configured project logger, optionally bound to *name*."""
    if not _CONFIGURED:
        configure_logger(
            log_file=os.getenv("LOG_FILE", "logs/traffic.log"),
            level=os.getenv("LOG_LEVEL", "INFO"),
        )
    return logger.bind(component=name) if name else logger
