# Filename: ui/milestone_tracker.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar

class MilestoneTracker(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        
        # Milestone Tracking
        self.milestone_label = QLabel("Milestone Tracker")
        self.milestone_progress = QProgressBar()
        self.milestone_progress.setValue(35)  # Sample progress

        # Layout setup
        self.layout.addWidget(self.milestone_label)
        self.layout.addWidget(QLabel("Current Milestone Progress"))
        self.layout.addWidget(self.milestone_progress)
        self.setLayout(self.layout)
