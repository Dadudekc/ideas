# notifier.py

import smtplib
from email.mime.text import MIMEText
from typing import Dict, Any
import logging

class Notifier:
    """
    Notifier class to send email notifications.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.logger = logger
        self.smtp_server = config.get("smtp_server")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.from_addr = config.get("from_addr")
        self.to_addr = config.get("to_addr")

    def send_email(self, subject: str, message: str):
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.from_addr
            msg['To'] = self.to_addr

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            self.logger.info(f"Email sent: {subject}")
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
