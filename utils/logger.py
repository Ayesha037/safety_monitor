from __future__ import annotations
import sys
from pathlib import Path
from loguru import logger

def setup_logger(log_level: str = "INFO", log_file: str = "logs/safety_monitor.log"):
    logger.remove()
    logger.add(
        sys.stdout, level=log_level, colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file, level="DEBUG", rotation="10 MB",
        retention="7 days", compression="zip", enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
    logger.info("Logger initialised — level={}", log_level)