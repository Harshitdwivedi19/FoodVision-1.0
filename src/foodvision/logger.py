"""
FoodVision 1.0 Production Logging System
Provides dual-destination logging with log rotation, structured formatting,
and contextual metadata.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Setup paths
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "foodvision.log"


def get_logger(name: str = "foodvision") -> logging.Logger:
    """
    Returns a configured logger with console and rotating file handlers.
    Max log size: 10MB, up to 5 backup logs kept.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Standard formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Rotating file handler (10 MB per file, max 5 files)
        file_handler = RotatingFileHandler(
            filename=str(LOG_FILE),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Prevent duplicate propagation
        logger.propagate = False

    return logger


logger = get_logger("foodvision")
logger.info("FoodVision 1.0 logger initialized. Logging to %s", LOG_FILE)
