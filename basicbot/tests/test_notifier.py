import unittest
from unittest.mock import patch, MagicMock
import logging
import smtplib
from basicbot.notifier import Notifier


class TestNotifier(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        self.config = {
            "smtp_server": "smtp.testserver.com",
            "smtp_port": 587,
            "username": "testuser",
            "password": "testpassword",
            "from_addr": "from@example.com",
            "to_addr": "to@example.com"
        }
        self.logger = logging.getLogger("TestNotifier")
        self.logger.setLevel(logging.DEBUG)  # Set level to capture logs
        # Remove all handlers associated with the logger
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        # Add a handler that does nothing to prevent actual logging
        self.logger.addHandler(logging.NullHandler())

        self.notifier = Notifier(self.config, self.logger)

    @patch("smtplib.SMTP", autospec=True)
    def test_send_email_success(self, mock_smtp):
        """
        Test successful email sending.
        """
        # Create a mock SMTP server instance
        mock_server = MagicMock()
        # Configure the context manager to return the mock_server
        mock_smtp.return_value.__enter__.return_value = mock_server

        subject = "Test Subject"
        message = "This is a test message."

        self.notifier.send_email(subject, message)

        # Verify SMTP is initialized correctly
        mock_smtp.assert_called_once_with(self.config["smtp_server"], self.config["smtp_port"])
        # Verify starttls is called
        mock_server.starttls.assert_called_once()
        # Verify login is called with correct credentials
        mock_server.login.assert_called_once_with(self.config["username"], self.config["password"])
        # Verify send_message is called once with a MIMEText message
        mock_server.send_message.assert_called_once()

    @patch("smtplib.SMTP", autospec=True)
    def test_send_email_failure(self, mock_smtp):
        """
        Test email sending failure.
        """
        # Create a mock SMTP server instance
        mock_server = MagicMock()
        # Configure the context manager to return the mock_server
        mock_smtp.return_value.__enter__.return_value = mock_server
        # Simulate an exception during login
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b'Authentication failed')

        subject = "Test Subject"
        message = "This is a test message."

        # Capture the log output
        with self.assertLogs(self.logger, level="ERROR") as log:
            self.notifier.send_email(subject, message)

        # Verify SMTP is initialized correctly
        mock_smtp.assert_called_once_with(self.config["smtp_server"], self.config["smtp_port"])
        # Verify starttls is called
        mock_server.starttls.assert_called_once()
        # Verify login is called with correct credentials
        mock_server.login.assert_called_once_with(self.config["username"], self.config["password"])
        # Verify send_message is NOT called due to failure
        mock_server.send_message.assert_not_called()
        # Verify that an error log was generated
        self.assertTrue(any("Failed to send email" in message for message in log.output))


if __name__ == "__main__":
    unittest.main()
