# Filename: notification_manager.py
# Description: Manages notifications for the Project Management App.

from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
import logging

# Configure logging for the NotificationManager
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class NotificationManager(QObject):
    """
    NotificationManager Class

    Handles notifications and reminders for the Project Management App. 
    Supports both immediate notifications and periodic reminders.
    """
    
    # Signal to notify other components when a new notification is received
    notification_received = pyqtSignal(str)

    def __init__(self, parent=None, reminder_interval: int = 3600000):
        """
        Initializes the notification manager with a parent QObject and optional reminder interval.

        Args:
            parent (QObject, optional): Parent object for the notification manager.
            reminder_interval (int, optional): Interval for reminders in milliseconds. Default is 1 hour.
        """
        super().__init__(parent)
        self.notifications = []  # Stores a log of notifications for tracking
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_reminder)
        self.timer.start(reminder_interval)  # Start periodic reminders
        
        logger.info("NotificationManager initialized with reminder interval set to every %d ms.", reminder_interval)

    def show_reminder(self):
        """Displays a periodic reminder message to the user."""
        reminder_message = "Reminder: Check project milestones and commit any updates."
        self.show_notification(reminder_message)
        logger.info("Reminder shown to the user: %s", reminder_message)

    def show_notification(self, message: str):
        """
        Displays a notification message using QMessageBox and logs it.

        Args:
            message (str): The notification message to display.
        """
        msg = QMessageBox(self.parent())
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Notification")
        msg.exec_()
        
        # Emit the notification signal and log the message
        self.notifications.append(message)
        self.notification_received.emit(message)
        logger.info("Notification displayed and signal emitted: %s", message)

    def send_notification(self, message: str):
        """
        Sends an on-demand notification to the user and logs it.

        Args:
            message (str): The notification message to send.
        """
        self.show_notification(message)
        logger.info("On-demand notification sent: %s", message)

    def stop_reminders(self):
        """Stops the periodic reminders."""
        self.timer.stop()
        logger.info("Periodic reminders stopped.")

    def start_reminders(self, interval: int):
        """
        Starts or restarts periodic reminders with a specified interval.

        Args:
            interval (int): Interval for reminders in milliseconds.
        """
        self.timer.start(interval)
        logger.info("Periodic reminders started with interval %d ms.", interval)

    def get_notification_log(self):
        """Returns the log of all sent notifications."""
        return self.notifications
