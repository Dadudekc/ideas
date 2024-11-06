# Filename: utils/git_commit_tracker.py

import subprocess
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

class GitCommitTracker:
    def __init__(self, parent, repo_path):
        self.parent = parent
        self.repo_path = repo_path
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_commit_reminder)
        self.timer.start(1800000)  # Reminder every 30 minutes

    def show_commit_reminder(self):
        msg = QMessageBox(self.parent)
        msg.setIcon(QMessageBox.Information)
        msg.setText("Remember to commit your changes if you've made significant progress.")
        msg.setWindowTitle("Git Commit Reminder")
        msg.exec_()

    def commit_changes(self, message):
        try:
            subprocess.run(["git", "-C", self.repo_path, "add", "."], check=True)
            subprocess.run(["git", "-C", self.repo_path, "commit", "-m", message], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Git commit failed: {e}")
