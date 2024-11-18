# -------------------------------------------------------------------
# File: config_handling/logging_setup.py
# Description: Configures logging with optional rotating file handler and
#              console output, supporting dynamic paths and feedback loop.
# -------------------------------------------------------------------

import logging
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler  # Correct import
import os
import sys

# Dynamically adjust project paths
script_dir = Path(__file__).resolve().parent
utilities_dir = script_dir.parent
sys.path.append(str(utilities_dir))

# Attempt to import ConfigManager with fallbacks
try:
    from config_handling.config_manager import ConfigManager
    print("ConfigManager imported from project structure")
except ModuleNotFoundError as e:
    try:
        from config_manager import ConfigManager
        print("ConfigManager imported as standalone")
    except ModuleNotFoundError:
        print(f"Warning: Could not import ConfigManager in either mode: {e}")


from logging.handlers import RotatingFileHandler  # Ensure proper import

def setup_logging(
    script_name="default_script",
    log_dir=None,
    max_log_size=5 * 1024 * 1024,
    backup_count=2,
    console_log_level=logging.INFO,
    file_log_level=logging.DEBUG,
    feedback_loop_enabled=False,
):
    """
    Sets up a portable logger with optional rotating file logging.

    Args:
        script_name (str): The name of the script for logger identification.
        log_dir (Path or str, optional): Directory for log storage.
        max_log_size (int, optional): Maximum size for log rotation (bytes).
        backup_count (int, optional): Number of rotated logs to retain.
        console_log_level (int): Logging level for console.
        file_log_level (int): Logging level for file output.
        feedback_loop_enabled (bool): Enables feedback tracking if needed.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Set the log directory based on environment variable or fallback to 'logs' in project root
    if log_dir is None:
        log_dir = Path(os.getenv('LOG_DIR', script_dir.parent / 'logs'))
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Construct the log file path
    log_file = log_dir / f"{script_name}.log"

    # Initialize the logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, filter per handler

    # Setup RotatingFileHandler for file logging with size-based rotation
    try:
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_log_size,
            backupCount=backup_count,
        )
        file_handler.setLevel(file_log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
    except Exception as e:
        print(f"Error setting up RotatingFileHandler: {e}")
        raise

    # Setup StreamHandler for console output
    try:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
    except Exception as e:
        print(f"Error setting up StreamHandler: {e}")
        raise

    # Attach handlers if not already added to prevent duplicates
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)

    # Enable feedback loop if specified
    if feedback_loop_enabled:
        enable_feedback_loop(logger)

    return logger

def enable_feedback_loop(logger):
    """
    Initializes a feedback loop for error/performance tracking in the logger.

    Args:
        logger (logging.Logger): The logger instance to add feedback loop.
    """
    logger.info("Feedback loop enabled for error/performance tracking.")

    def log_feedback(message, level=logging.INFO):
        # Send critical logs to external systems if desired
        if level == logging.ERROR:
            # Placeholder for external integration
            pass
        logger.log(level, message)

    # Attach the feedback function to the logger
    logger.feedback = log_feedback


def debug_logging_setup(logger, log_file):
    """
    Verifies logging setup by testing log levels and checking file existence.

    Args:
        logger (logging.Logger): Logger instance to test.
        log_file (Path): Expected file path for log output.
    """
    try:
        # Test different log levels
        logger.debug("Testing debug level logging.")
        logger.info("Testing info level logging.")
        logger.warning("Testing warning level logging.")
        logger.error("Testing error level logging.")
        logger.critical("Testing critical level logging.")

        # Confirm log file path and handler types
        log_path = Path(log_file)
        print(f"Log file exists: {log_path.exists()}")
        print(f"Handlers attached to logger: {[type(h) for h in logger.handlers]}")

    except Exception as e:
        print(f"Error during logging setup debug: {e}")


# Example usage
if __name__ == "__main__":
    logger = setup_logging(
        script_name="standalone_trading_robot",
        log_dir="C:/Projects/#TODO/ideas/_passive_income/ArmyOfRobots/My1strobot/logs"
    )
    logger.info("Test log message to verify logging setup.")
