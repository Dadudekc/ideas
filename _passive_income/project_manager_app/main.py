# Filename: main.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from ui.dashboard import Dashboard
from ui.task_board import TaskBoard
from ui.milestone_tracker import MilestoneTracker
from ui.analytics import Analytics
from ui.notifications import NotificationManager

class ProjectManagerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Project Management App")
        self.setGeometry(100, 100, 1200, 800)

        # Setup tabs for each module
        self.tabs = QTabWidget()
        self.tabs.addTab(Dashboard(), "Dashboard")
        self.tabs.addTab(TaskBoard(), "Task Board")
        self.tabs.addTab(MilestoneTracker(), "Milestone Tracker")
        self.tabs.addTab(Analytics(), "Analytics")

        # Setup main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)

        # Initialize notifications
        NotificationManager(self)

        # Set layout to the main window
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProjectManagerApp()
    window.show()
    sys.exit(app.exec_())
