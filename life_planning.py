import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox,
    QPushButton, QTextEdit, QListWidget, QHBoxLayout, QInputDialog
)


# Task Manager class to handle task storage and updates
class TaskManager:
    def __init__(self):
        self.tasks = {
            "Budget": [],
            "Career Goals": [],
            "Development Project Goals": [],
            "Cities Skylines": [],
            "OSRS Goals": [],
            "Day Trading": [],
            "Digital Dreamscape": []
        }
        self.load_tasks()

    def load_tasks(self):
        try:
            with open('tasks.json', 'r') as f:
                self.tasks = json.load(f)
        except FileNotFoundError:
            pass

    def save_tasks(self):
        with open('tasks.json', 'w') as f:
            json.dump(self.tasks, f, indent=4)

    def add_task(self, category, task):
        self.tasks[category].append(task)
        self.save_tasks()

    def remove_task(self, category, task):
        if task in self.tasks[category]:
            self.tasks[category].remove(task)
            self.save_tasks()


# Main application class
class PlanningApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Life Planning System")
        self.layout = QVBoxLayout()
        self.task_manager = TaskManager()

        self.category_label = QLabel("Select Category:")
        self.layout.addWidget(self.category_label)

        # Dropdown for categories
        self.category_dropdown = QComboBox()
        self.category_dropdown.addItems(self.task_manager.tasks.keys())
        self.layout.addWidget(self.category_dropdown)

        # Task list
        self.task_list = QListWidget()
        self.layout.addWidget(self.task_list)

        # Buttons to add/remove tasks
        self.button_layout = QHBoxLayout()

        self.add_task_button = QPushButton("Add Task")
        self.add_task_button.clicked.connect(self.add_task)
        self.button_layout.addWidget(self.add_task_button)

        self.remove_task_button = QPushButton("Remove Task")
        self.remove_task_button.clicked.connect(self.remove_task)
        self.button_layout.addWidget(self.remove_task_button)

        self.layout.addLayout(self.button_layout)

        # Task details section
        self.task_detail_label = QLabel("Task Details:")
        self.layout.addWidget(self.task_detail_label)
        self.task_detail_text = QTextEdit()
        self.layout.addWidget(self.task_detail_text)

        # Load initial tasks
        self.category_dropdown.currentIndexChanged.connect(self.load_tasks)
        self.load_tasks()

        self.setLayout(self.layout)

    def load_tasks(self):
        category = self.category_dropdown.currentText()
        self.task_list.clear()
        for task in self.task_manager.tasks[category]:
            self.task_list.addItem(task)

    def add_task(self):
        category = self.category_dropdown.currentText()
        task, ok = QInputDialog.getText(self, "Add Task", "Enter the task:")
        if ok and task:
            self.task_manager.add_task(category, task)
            self.load_tasks()

    def remove_task(self):
        category = self.category_dropdown.currentText()
        selected_task = self.task_list.currentItem()
        if selected_task:
            self.task_manager.remove_task(category, selected_task.text())
            self.load_tasks()


# Main function to run the app
def main():
    app = QApplication(sys.argv)
    window = PlanningApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
