import sys
import json
import requests  # To send HTTP requests to Ollama's API
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, 
    QLineEdit, QTextEdit, QListWidget, QComboBox, QMessageBox, QInputDialog
)

# Function to query Mistral model via Ollama's HTTP API
def query_mistral_via_ollama(text):
    try:
        url = "http://localhost:11434/chat"  # Replace with your Ollama API endpoint if different
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "mistral",
            "messages": [
                {"role": "user", "content": text}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        print(f"Ollama Response: {data}")  # Debugging: Print the response from Ollama

        if isinstance(data, dict):
            if 'message' in data and 'content' in data['message']:
                return data['message']['content']
            elif 'choices' in data and len(data['choices']) > 0 and 'text' in data['choices'][0]:
                return data['choices'][0]['text'].strip()
            elif 'text' in data:
                return data['text']
            else:
                print("Unexpected response format from Ollama.")
                return None
        elif isinstance(data, str):
            return data
        else:
            print("Unexpected response type from Ollama.")
            return None
    except Exception as e:
        print(f"Error querying Mistral via Ollama: {e}")
        traceback.print_exc()
        return None

class ProjectManager:
    def __init__(self, filename='projects.json'):
        self.filename = filename
        self.tasks = []
        self.load_tasks()

    def load_tasks(self):
        try:
            with open(self.filename, 'r') as f:
                self.tasks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.tasks = []

    def save_tasks(self):
        with open(self.filename, 'w') as f:
            json.dump(self.tasks, f, indent=4)

    def add_task(self, title, category, description, priority):
        task = {
            'title': title,
            'category': category,
            'description': description,
            'priority': priority,
            'completed': False
        }
        self.tasks.append(task)
        self.save_tasks()

    def remove_task(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks.pop(index)
            self.save_tasks()

    def get_tasks(self):
        return self.tasks

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project Organizer")
        self.setGeometry(100, 100, 600, 400)
        self.project_manager = ProjectManager()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title Input
        self.title_input = QLineEdit(self)
        self.title_input.setPlaceholderText("Enter task title")
        layout.addWidget(self.title_input)

        # Category Dropdown (Project-related categories)
        self.category_combo = QComboBox(self)
        self.category_combo.addItems([
            "Research", "Development", "Testing", "Documentation", 
            "Deployment", "Maintenance", "Design", "Miscellaneous"
        ])
        layout.addWidget(self.category_combo)

        # Description Input
        self.description_input = QTextEdit(self)
        self.description_input.setPlaceholderText("Enter task description")
        layout.addWidget(self.description_input)

        # Priority ComboBox
        self.priority_combo = QComboBox(self)
        self.priority_combo.addItems(["Low", "Medium", "High"])
        layout.addWidget(self.priority_combo)

        # Add Button
        add_button = QPushButton("Add Task", self)
        add_button.clicked.connect(self.add_task)
        layout.addWidget(add_button)

        # Task List
        self.task_list = QListWidget(self)
        layout.addWidget(self.task_list)

        # Delete Button
        delete_button = QPushButton("Delete Selected Task", self)
        delete_button.clicked.connect(self.delete_task)
        layout.addWidget(delete_button)

        # NLP Button (Mistral via Ollama)
        nlp_button = QPushButton("Add Task via NLP", self)
        nlp_button.clicked.connect(self.add_task_nlp)
        layout.addWidget(nlp_button)

        # Set main layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Load and display tasks
        self.load_tasks()

    def add_task(self):
        title = self.title_input.text()
        category = self.category_combo.currentText()  # Get the selected category from the dropdown
        description = self.description_input.toPlainText()
        priority = self.priority_combo.currentText()

        if not title or not category:
            QMessageBox.warning(self, "Input Error", "Title and Category are required!")
            return

        self.project_manager.add_task(title, category, description, priority)
        self.load_tasks()

        self.title_input.clear()
        self.description_input.clear()

    def load_tasks(self):
        self.task_list.clear()
        tasks = self.project_manager.get_tasks()
        for task in tasks:
            display_text = f"{task['title']} - {task['category']} - {task['priority']} (Completed: {task['completed']})"
            self.task_list.addItem(display_text)

    def delete_task(self):
        selected_task_index = self.task_list.currentRow()
        if selected_task_index != -1:
            self.project_manager.remove_task(selected_task_index)
            self.load_tasks()

    def add_task_nlp(self):
        user_input, ok = QInputDialog.getText(self, "NLP Task", "Describe your task:")
        if ok and user_input:
            result = query_mistral_via_ollama(user_input)
            if result:
                title = result  # Assuming result is a plain string returned by Ollama
                category = self.category_combo.currentText()  # Use the selected category
                description = ""
                priority = self.priority_combo.currentText()

                self.project_manager.add_task(title, category, description, priority)
                self.load_tasks()
            else:
                QMessageBox.warning(self, "NLP Error", "Failed to process NLP input.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
