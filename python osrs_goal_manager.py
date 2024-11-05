import sys
import json
import ollama  # Import Ollama for model interaction
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QTextEdit, QListWidget, QComboBox, QMessageBox, QInputDialog
import requests

# Function to query Mistral model via Ollama's HTTP API
def query_mistral_via_ollama(text):
    try:
        url = "http://localhost:11434/chat"  # Replace with your Ollama API endpoint if different
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "mistral",  # Ensure this is a string, not an object
            "messages": [        # The messages should be wrapped in a list
                {"role": "user", "content": text}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        print(f"Ollama Response: {data}")  # Debugging: Print the response from Ollama

        # Handle different possible response structures
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
        traceback.print_exc()  # Print the full traceback of the error
        return None


class GoalManager:
    def __init__(self, filename='goals.json'):
        self.filename = filename
        self.goals = []
        self.load_goals()

    def load_goals(self):
        try:
            with open(self.filename, 'r') as f:
                self.goals = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.goals = []

    def save_goals(self):
        with open(self.filename, 'w') as f:
            json.dump(self.goals, f, indent=4)

    def add_goal(self, title, category, description, priority):
        goal = {
            'title': title,
            'category': category,
            'description': description,
            'priority': priority,
            'completed': False
        }
        self.goals.append(goal)
        self.save_goals()

    def remove_goal(self, index):
        if 0 <= index < len(self.goals):
            self.goals.pop(index)
            self.save_goals()

    def get_goals(self):
        return self.goals


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OSRS Goals Manager")
        self.setGeometry(100, 100, 600, 400)
        self.goal_manager = GoalManager()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title Input
        self.title_input = QLineEdit(self)
        self.title_input.setPlaceholderText("Enter goal title")
        layout.addWidget(self.title_input)

        # Category Dropdown (Predefined OSRS categories)
        self.category_combo = QComboBox(self)
        self.category_combo.addItems(["Quest", "Skill", "Combat", "Minigame", "Collection", "Boss", "Achievement Diary", "Miscellaneous"])
        layout.addWidget(self.category_combo)

        # Description Input
        self.description_input = QTextEdit(self)
        self.description_input.setPlaceholderText("Enter goal description")
        layout.addWidget(self.description_input)

        # Priority ComboBox
        self.priority_combo = QComboBox(self)
        self.priority_combo.addItems(["Low", "Medium", "High"])
        layout.addWidget(self.priority_combo)

        # Add Button
        add_button = QPushButton("Add Goal", self)
        add_button.clicked.connect(self.add_goal)
        layout.addWidget(add_button)

        # Goals List
        self.goals_list = QListWidget(self)
        layout.addWidget(self.goals_list)

        # Delete Button
        delete_button = QPushButton("Delete Selected Goal", self)
        delete_button.clicked.connect(self.delete_goal)
        layout.addWidget(delete_button)

        # NLP Button (Mistral via Ollama)
        nlp_button = QPushButton("Add Goal via NLP", self)
        nlp_button.clicked.connect(self.add_goal_nlp)
        layout.addWidget(nlp_button)

        # Set main layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Load and display goals
        self.load_goals()

    def add_goal(self):
        title = self.title_input.text()
        category = self.category_combo.currentText()  # Get the selected category from the dropdown
        description = self.description_input.toPlainText()
        priority = self.priority_combo.currentText()

        if not title or not category:
            QMessageBox.warning(self, "Input Error", "Title and Category are required!")
            return

        self.goal_manager.add_goal(title, category, description, priority)
        self.load_goals()

        self.title_input.clear()
        self.description_input.clear()

    def load_goals(self):
        self.goals_list.clear()
        goals = self.goal_manager.get_goals()
        for goal in goals:
            display_text = f"{goal['title']} - {goal['category']} - {goal['priority']} (Completed: {goal['completed']})"
            self.goals_list.addItem(display_text)

    def delete_goal(self):
        selected_goal_index = self.goals_list.currentRow()
        if selected_goal_index != -1:
            self.goal_manager.remove_goal(selected_goal_index)
            self.load_goals()

    def add_goal_nlp(self):
        user_input, ok = QInputDialog.getText(self, "NLP Goal", "Describe your goal:")
        if ok and user_input:
            result = query_mistral_via_ollama(user_input)
            if result:
                title = result  # Assuming result is a plain string returned by Ollama
                category = self.category_combo.currentText()  # Use the selected category
                description = ""
                priority = self.priority_combo.currentText()

                self.goal_manager.add_goal(title, category, description, priority)
                self.load_goals()
            else:
                QMessageBox.warning(self, "NLP Error", "Failed to process NLP input.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
