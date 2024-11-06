# Filename: ui/dashboard.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QProgressBar
from models.analytics_model import AnalyticsModel

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        # Project KPI Summary
        self.kpi_label = QLabel("Project KPIs")
        self.kpi_list = QListWidget()

        # Add progress bars and KPIs
        self.task_progress = QProgressBar()
        self.task_progress.setValue(40)  # Example progress
        self.milestone_progress = QProgressBar()
        self.milestone_progress.setValue(25)  # Example progress
        
        # Analytics summary
        self.analytics_summary = QLabel(AnalyticsModel().get_summary())  # Displays key metrics like tasks completed

        # Layout arrangement
        self.layout.addWidget(self.kpi_label)
        self.layout.addWidget(self.kpi_list)
        self.layout.addWidget(QLabel("Overall Task Progress"))
        self.layout.addWidget(self.task_progress)
        self.layout.addWidget(QLabel("Milestone Completion"))
        self.layout.addWidget(self.milestone_progress)
        self.layout.addWidget(self.analytics_summary)
        self.setLayout(self.layout)
