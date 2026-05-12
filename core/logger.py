"""Structured logging setup for AI Research Assistant."""

import os
import sys
from pathlib import Path
from loguru import logger
from core.config import config


def setup_logging():
    """Configure structured logging with file and console output."""
    
    # Remove default logger
    logger.remove()
    
    # Ensure logs directory exists
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Console logging with colors
    logger.add(
        sys.stdout,
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File logging with rotation
    logger.add(
        config.log_file,
        level=config.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    return logger


# Initialize logger
app_logger = setup_logging()

# Export for easy import
__all__ = ["app_logger"]