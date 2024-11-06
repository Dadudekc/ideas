# Filename: notifications.py
# Handles notifications and reminders.

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

class NotificationManager:
    def __init__(self, parent):
        self.parent = parent
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_notification)
        self.timer.start(3600000)  # Notify every hour

    def show_notification(self):
        msg = QMessageBox(self.parent)
        msg.setIcon(QMessageBox.Information)
        msg.setText("Reminder: Check project milestones and commit any updates.")
        msg.setWindowTitle("Notification")
        msg.exec_()
