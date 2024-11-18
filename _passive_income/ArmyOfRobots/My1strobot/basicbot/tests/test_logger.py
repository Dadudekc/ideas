import unittest
import logging
import os
from pathlib import Path
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

    def tearDown(self):
        """
        Clean up after each test.
        """
        logger = logging.getLogger(self.logger_name)
        # Close all handlers
        while logger.handlers:
            handler = logger.handlers[0]
            handler.close()
            logger.removeHandler(handler)

        # Remove log files
        for file in Path(self.config.log_dir).glob("test_log*"):
            file.unlink()
            
    @patch("logging.StreamHandler.emit")
    def test_console_logging(self, mock_emit):
        """
        Test that logs are emitted to the console.
        """
        logger = setup_logger(self.logger_name, self.config)
        logger.info("Test console log message.")

        mock_emit.assert_called()

def test_logger_file_rotation(self):
    """
    Test file rotation by writing more than max_log_size.
    """
    logger = setup_logger(self.logger_name, self.config)

    # Write logs to exceed the maxBytes threshold
    for _ in range(2000):  # Adjusted for quicker execution
        logger.debug("This is a debug log for testing file rotation.")

    # Allow time for the rotation to occur (if necessary)
    import time
    time.sleep(1)

    # Flush and close handlers to release file locks
    for handler in logger.handlers:
        handler.flush()
        handler.close()

    # Check log files in the directory
    log_files = list(Path(self.config.log_dir).glob("test_log*"))

    # Assert at least 1 original and 2 backups exist
    self.assertGreaterEqual(len(log_files), 3)  # 1 original + 2 backups
    for file in log_files:
        self.assertGreater(file.stat().st_size, 0)  # Ensure files are not empty



if __name__ == "__main__":
    unittest.main()
