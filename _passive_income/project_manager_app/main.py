# Filename: main.py
# Description: Main entry point for the Advanced Project Management App, initializing the GUI with multiple tabs for different project management functionalities.

import sys
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import Qt
from ui.dashboard import Dashboard
from ui.task_board import TaskBoard
from ui.milestone_tracker import MilestoneTracker
from ui.analytics import Analytics
from ui.notifications import NotificationManager
from ui.settings import Settings

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ProjectManagerApp(QMainWindow):
    """
    ProjectManagerApp Class

    A PyQt5 application for managing projects, tasks, milestones, and analytics.
    This main window initializes different tabs for various functionalities.
    """

    def __init__(self):
        """Initializes the main project manager application window."""
        super().__init__()
        try:
            self._setup_window()
            self._initialize_tabs()
            self._initialize_notifications()
            self._initialize_status_bar()
            logger.info("Project Manager App initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize the application: {e}")
            self._show_error("Initialization Error", str(e))

    def _setup_window(self):
        """Configures main window settings like title, size, and layout."""
        self.setWindowTitle("Advanced Project Management App")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        logger.info("Window setup completed.")

    def _initialize_tabs(self):
        """Initializes and adds the main application tabs."""
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(False)  # Makes tabs non-closable by default

        # Initialize core tabs
        self.tabs.addTab(Dashboard(), "Dashboard")
        self.tabs.addTab(TaskBoard(), "Task Board")
        self.tabs.addTab(MilestoneTracker(), "Milestone Tracker")
        self.tabs.addTab(Analytics(), "Analytics")
        self.tabs.addTab(Settings(), "Settings")  # New settings tab

        # Set up the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)

        # Set up the container and layout for the main window
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        logger.info("Tabs initialized successfully.")

    def _initialize_notifications(self):
        """Initializes the notification manager for app notifications."""
        self.notification_manager = NotificationManager(self)
        self.notification_manager.notification_received.connect(self._display_notification)
        logger.info("Notification manager initialized.")

    def _initialize_status_bar(self):
        """Sets up the status bar for displaying messages."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Welcome to the Project Management App!")
        logger.info("Status bar initialized.")

    def _display_notification(self, message):
        """
        Display notifications in both the status bar and a pop-up if critical.

        Args:
            message (str): The notification message to display.
        """
        self.status_bar.showMessage(message)
        if "Critical" in message:
            QMessageBox.warning(self, "Critical Notification", message)
        logger.info(f"Notification displayed: {message}")

    def add_custom_tab(self, widget: QWidget, title: str):
        """
        Adds a custom tab to the main window, allowing for dynamic expansion.

        Args:
            widget (QWidget): The widget for the new tab content.
            title (str): Title of the new tab.
        """
        self.tabs.addTab(widget, title)
        logger.info(f"Custom tab added: {title}")

    def remove_tab(self, index: int):
        """
        Removes a tab from the main window.

        Args:
            index (int): Index of the tab to remove.
        """
        if 0 <= index < self.tabs.count():
            tab_title = self.tabs.tabText(index)
            self.tabs.removeTab(index)
            logger.info(f"Removed tab: {tab_title}")
        else:
            logger.warning(f"Invalid tab index: {index}")

    def _show_error(self, title, message):
        """Displays an error message in a message box."""
        QMessageBox.critical(self, title, message)
        logger.error(f"{title}: {message}")

    def closeEvent(self, event):
        """Handles the close event for clean shutdown."""
        reply = QMessageBox.question(
            self, "Exit", "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
            logger.info("Application closed by user.")
        else:
            event.ignore()
            logger.info("Application exit canceled by user.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProjectManagerApp()
    window.show()
    sys.exit(app.exec_())
