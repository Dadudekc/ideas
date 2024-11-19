import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent  # Adjust as needed
sys.path.insert(0, str(project_root))

import unittest
import logging
import os
from unittest.mock import patch
from basicbot.logger import setup_logger
from basicbot.config import LoggingConfig


class TestLogger(unittest.TestCase):
    """
    Unit tests for the logger setup.
    """

    def setUp(self):
        """
        Set up a temporary LoggingConfig for testing.
        """
        self.config = LoggingConfig(
            log_dir="./test_logs",
            log_level="DEBUG",
            log_file="test_log.log",
            max_log_size=1024,  # 1 KB for testing rotation
            backup_count=2
        )
        self.logger_name = "TestLogger"
        self.logger = setup_logger(self.logger_name, self.config)  # Pass LoggingConfig


    def tearDown(self):
        """
        Clean up after each test.
        """
        logger = logging.getLogger(self.logger_name)
        # Close all handlers
        while logger.handlers:
            handler = logger.handlers.pop()
            handler.close()

        # Remove log files
        log_dir = Path(self.config.log_dir)
        if log_dir.exists():
            for file in log_dir.glob("test_log*"):
                file.unlink()

    @patch("logging.StreamHandler.emit")
    def test_console_logging(self, mock_emit):
        """
        Test that logs are emitted to the console.
        """
        self.logger.info("Test console log message.")
        mock_emit.assert_called()

    def test_logger_file_rotation(self):
        """
        Test file rotation by writing more than max_log_size.
        """
        # Write logs to exceed the maxBytes threshold
        for _ in range(1000):  # Adjusted for quicker execution
            self.logger.debug("This is a debug log for testing file rotation.")

        # Ensure all handlers are flushed and closed
        logger = logging.getLogger(self.logger_name)
        for handler in logger.handlers:
            handler.flush()
            handler.close()

        # Check log files in the directory
        log_dir = Path(self.config.log_dir)
        log_files = list(log_dir.glob("test_log*"))

        # Assert at least 1 original and 2 backups exist
        self.assertGreaterEqual(len(log_files), 3, "Log rotation did not create sufficient backups.")
        for file in log_files:
            self.assertGreater(file.stat().st_size, 0, "Log file should not be empty.")


if __name__ == "__main__":
    unittest.main()
