# Filename: analytics.py
# Provides project analytics such as time tracking, completion metrics.

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class Analytics(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.analytics_label = QLabel("Analytics & Metrics")
        self.analytics_overview = QLabel("Time spent: 5 hrs\nTasks Completed: 8\nEfficiency: 80%")

        self.layout.addWidget(self.analytics_label)
        self.layout.addWidget(self.analytics_overview)
        self.setLayout(self.layout)
