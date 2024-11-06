# Filename: ui/task_board.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem

class TaskBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        
        # Task Board Title
        self.task_board_label = QLabel("Task Board - Priority and Status")
        self.task_table = QTableWidget(0, 3)  # Rows, Columns: Task, Priority, Status
        self.task_table.setHorizontalHeaderLabels(["Task", "Priority", "Status"])

        # Populate sample tasks
        self.add_task("Develop AI Utility Module", "High", "In Progress")
        self.add_task("Refactor Data Fetcher", "Medium", "Pending")

        # Add widgets to layout
        self.layout.addWidget(self.task_board_label)
        self.layout.addWidget(self.task_table)
        self.setLayout(self.layout)

    def add_task(self, name, priority, status):
        row = self.task_table.rowCount()
        self.task_table.insertRow(row)
        self.task_table.setItem(row, 0, QTableWidgetItem(name))
        self.task_table.setItem(row, 1, QTableWidgetItem(priority))
        self.task_table.setItem(row, 2, QTableWidgetItem(status))
