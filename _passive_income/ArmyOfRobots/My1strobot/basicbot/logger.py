import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from basicbot.config import LoggingConfig

def setup_logger(name: str, config: LoggingConfig) -> logging.Logger:
    """
    Set up a logger that logs to both console and file with rotation.
    """
    logger = logging.getLogger(name)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Explicitly set the logger level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Create log directory if it doesn't exist
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        Path(config.log_dir) / config.log_file,
        maxBytes=config.max_log_size,
        backupCount=config.backup_count
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

