# Filename: ui/task_board.py
# Description: Comprehensive Task Management Board with real-time tracking, priority styling, filtering, and deadline tracking.

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QLineEdit, QMenu, QAction, QMessageBox
)
from PyQt5.QtCore import Qt, QDateTime, QTimer
from PyQt5.QtGui import QColor

import time


class TaskBoard(QWidget):
    """
    TaskBoard Class
    
    An advanced Task Management Board with real-time tracking, detailed control, priority styling,
    filtering, and deadline tracking.
    """

    PRIORITY_MAP = {"High": 1, "Medium": 2, "Low": 3}
    STATUS_OPTIONS = ["Not Started", "In Progress", "Completed"]

    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.task_tracker.start(1000)  # Update task durations every second

    def _setup_ui(self):
        """Initializes and sets up the UI layout for the TaskBoard."""
        self.layout = QVBoxLayout()

        # Title
        self.task_board_label = QLabel("Task Board - Project Phases and Real-Time Tracking")
        self.task_board_label.setAlignment(Qt.AlignCenter)
        self.task_board_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0055A5; margin-bottom: 10px;")
        self.layout.addWidget(self.task_board_label)

        # Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search tasks by name, priority, status, or project phase...")
        self.search_bar.textChanged.connect(self.filter_tasks)
        self.layout.addWidget(self.search_bar)

        # Task Table Setup
        self.task_table = QTableWidget(0, 6)  # Columns: Task, Priority, Status, Start Time, Duration, Deadline
        self.task_table.setHorizontalHeaderLabels(["Task", "Priority", "Status", "Start Time", "Duration", "Deadline"])
        self.task_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.task_table.setSortingEnabled(True)
        self.task_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.task_table.customContextMenuRequested.connect(self.open_context_menu)
        self.layout.addWidget(self.task_table)

        # Timer for updating task durations
        self.task_tracker = QTimer(self)
        self.task_tracker.timeout.connect(self.update_task_durations)

        self.setLayout(self.layout)

    def add_task(self, name, priority, status="Not Started", start_time=None, duration=None, deadline=None):
        """Adds a new task with details and tracks start time and duration."""
        row = self.task_table.rowCount()
        self.task_table.insertRow(row)

        start_time = start_time or QDateTime.currentDateTime()
        duration = duration or "00:00:00"
        deadline = deadline or "No Deadline Set"

        task_item = QTableWidgetItem(name)
        priority_item = QTableWidgetItem(priority)
        status_item = QTableWidgetItem(status)
        start_time_item = QTableWidgetItem(start_time.toString("yyyy-MM-dd HH:mm:ss"))
        duration_item = QTableWidgetItem(duration)
        deadline_item = QTableWidgetItem(deadline)

        for item in (task_item, priority_item, status_item, start_time_item, duration_item, deadline_item):
            item.setTextAlignment(Qt.AlignCenter)

        self.task_table.setItem(row, 0, task_item)
        self.task_table.setItem(row, 1, priority_item)
        self.task_table.setItem(row, 2, status_item)
        self.task_table.setItem(row, 3, start_time_item)
        self.task_table.setItem(row, 4, duration_item)
        self.task_table.setItem(row, 5, deadline_item)

        self.apply_priority_styling(priority_item, priority)

    def apply_priority_styling(self, item, priority):
        """Styles priority cells with color coding based on priority level."""
        color_map = {"High": "#FF6666", "Medium": "#FFCC66", "Low": "#66FF66"}
        item.setBackground(QColor(color_map.get(priority, "#FFFFFF")))

    def update_task_durations(self):
        """Updates duration for tasks in progress."""
        for row in range(self.task_table.rowCount()):
            status_item = self.task_table.item(row, 2)
            if status_item and status_item.text() == "In Progress":
                start_time_str = self.task_table.item(row, 3).text()
                start_time = QDateTime.fromString(start_time_str, "yyyy-MM-dd HH:mm:ss")
                elapsed_time = start_time.secsTo(QDateTime.currentDateTime())
                duration_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                self.task_table.setItem(row, 4, QTableWidgetItem(duration_str))

    def filter_tasks(self):
        """Filters tasks based on search query from the search bar."""
        query = self.search_bar.text().lower()
        for row in range(self.task_table.rowCount()):
            match = any(query in self.task_table.item(row, col).text().lower() for col in range(self.task_table.columnCount()))
            self.task_table.setRowHidden(row, not match)

    def open_context_menu(self, position):
        """Opens a context menu with task options like complete, edit, delete, and set deadline."""
        menu = QMenu()

        mark_completed = QAction("Mark as Completed", self)
        mark_completed.triggered.connect(lambda: self.update_task_status(self.task_table.currentRow(), "Completed"))
        menu.addAction(mark_completed)

        edit_task_action = QAction("Edit Task", self)
        edit_task_action.triggered.connect(lambda: self.edit_task(self.task_table.currentRow()))
        menu.addAction(edit_task_action)

        set_deadline_action = QAction("Set Deadline", self)
        set_deadline_action.triggered.connect(lambda: self.set_task_deadline(self.task_table.currentRow()))
        menu.addAction(set_deadline_action)

        delete_task_action = QAction("Delete Task", self)
        delete_task_action.triggered.connect(lambda: self.delete_task(self.task_table.currentRow()))
        menu.addAction(delete_task_action)

        menu.exec_(self.task_table.viewport().mapToGlobal(position))

    def update_task_status(self, row, status):
        """Updates the status of a task and adjusts the UI accordingly."""
        if row >= 0:
            self.task_table.setItem(row, 2, QTableWidgetItem(status))

    def edit_task(self, row):
        """Opens an editor for modifying task details."""
        if row >= 0:
            task_name = self.task_table.item(row, 0).text()
            QMessageBox.information(self, "Edit Task", f"Editing task: {task_name}")

    def set_task_deadline(self, row):
        """Sets or updates a task deadline."""
        if row >= 0:
            current_deadline = self.task_table.item(row, 5).text()
            new_deadline, confirmed = QInputDialog.getText(self, "Set Deadline", f"Current deadline: {current_deadline}\nEnter new deadline (YYYY-MM-DD HH:MM:SS):")
            if confirmed:
                self.task_table.setItem(row, 5, QTableWidgetItem(new_deadline))

    def delete_task(self, row):
        """Removes a task from the task board."""
        if row >= 0:
            task_name = self.task_table.item(row, 0).text()
            confirmation = QMessageBox.question(self, "Delete Task", f"Are you sure you want to delete '{task_name}'?")
            if confirmation == QMessageBox.Yes:
                self.task_table.removeRow(row)
