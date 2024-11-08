# utils/logger.py
import logging
from config import Config

def setup_logger():
    logger = logging.getLogger("TSLA_Price_Monitor")
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()
