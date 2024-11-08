# gui/main_window.py
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHeaderView,
    QLineEdit,
    QMenu,
    QAction,
    QMessageBox,
    QHBoxLayout,
    QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5 import QtGui
from typing import Optional, Dict, Any
from integrations.alpaca_integration import AlpacaIntegration
from integrations.ollama_integration import OllamaIntegration
from models.database import Project, Insight
from sqlalchemy.orm import Session
from utils.logger import logger
from utils.data_fetcher import DataFetcher
import matplotlib.pyplot as plt

class TaskBoard(QWidget):
    """
    TaskBoard Class

    Displays a dynamic and interactive list of tasks with options for filtering, sorting, editing, 
    and priority-based color-coding.
    """

    PRIORITY_MAP = {"High": 1, "Medium": 2, "Low": 3}

    def __init__(self, session: Session, alpaca: AlpacaIntegration):
        super().__init__()
        self.session = session
        self.alpaca = alpaca
        self.ollama = OllamaIntegration()
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.task_board_label = QLabel("Task Board - Manage and Track Your Tasks")
        self.task_board_label.setAlignment(Qt.AlignCenter)
        self.task_board_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0055A5; margin-bottom: 15px;")
        self.layout.addWidget(self.task_board_label)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search tasks by name, priority, or status...")
        self.search_bar.textChanged.connect(self.filter_tasks)
        self.layout.addWidget(self.search_bar)

        self.task_table = QTableWidget(0, 3)
        self.task_table.setHorizontalHeaderLabels(["Task", "Priority", "Status"])
        self.task_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.task_table.setSortingEnabled(True)
        self.task_table.itemDoubleClicked.connect(self.edit_task)
        self.task_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.task_table.customContextMenuRequested.connect(self.open_context_menu)
        self.layout.addWidget(self.task_table)

        self.init_buttons()

        # Visualization Widget
        self.price_chart_button = QPushButton("Show TSLA Price Chart")
        self.price_chart_button.clicked.connect(self.show_price_chart)
        self.layout.addWidget(self.price_chart_button)

        self.setLayout(self.layout)
        self.load_tasks_from_db()

    def init_buttons(self):
        self.add_task_button = QPushButton("Add Task")
        self.add_task_button.clicked.connect(self.add_task_dialog)

        self.generate_insight_button = QPushButton("Generate Insight for TSLA")
        self.generate_insight_button.clicked.connect(self.generate_insight_for_price_action)

        self.alpaca_info_button = QPushButton("View Alpaca Account Info")
        self.alpaca_info_button.clicked.connect(self.view_alpaca_account_info)

        self.alpaca_fetch_button = QPushButton("Fetch TSLA Stock Data from Alpaca")
        self.alpaca_fetch_button.clicked.connect(self.fetch_alpaca_stock_data)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_task_button)
        button_layout.addWidget(self.generate_insight_button)
        button_layout.addWidget(self.alpaca_info_button)
        button_layout.addWidget(self.alpaca_fetch_button)
        self.layout.addLayout(button_layout)

    def load_tasks_from_db(self):
        """Loads tasks from the database into the table."""
        projects = self.session.query(Project).all()
        for project in projects:
            for insight in project.insights:
                self.add_task(insight.content, self.get_priority_label(insight.id), "Pending")

    def get_priority_label(self, priority: int) -> str:
        """Returns the priority label based on the priority value."""
        for label, value in self.PRIORITY_MAP.items():
            if value == priority:
                return label
        return "Low"

    def add_task(self, name: str, priority: str, status: str):
        """Adds a new task to the task table with color-coded priority."""
        row = self.task_table.rowCount()
        self.task_table.insertRow(row)

        # Create items with tooltip and center alignment
        task_item = QTableWidgetItem(name)
        task_item.setToolTip(f"Task: {name}")
        priority_item = QTableWidgetItem(priority)
        priority_item.setToolTip(f"Priority: {priority}")
        status_item = QTableWidgetItem(status)
        status_item.setToolTip(f"Status: {status}")

        for item in (task_item, priority_item, status_item):
            item.setTextAlignment(Qt.AlignCenter)

        # Apply color-coding based on priority
        self.apply_priority_styling(priority_item, priority)

        # Add items to the table
        self.task_table.setItem(row, 0, task_item)
        self.task_table.setItem(row, 1, priority_item)
        self.task_table.setItem(row, 2, status_item)

    def apply_priority_styling(self, item: QTableWidgetItem, priority: str):
        """Applies color styling based on task priority."""
        color = {"High": "#FF6666", "Medium": "#FFCC66", "Low": "#66FF66"}.get(priority, "#FFFFFF")
        item.setBackground(QtGui.QColor(color))

    def edit_task(self, item: QTableWidgetItem):
        """Enables editing of tasks on double-click."""
        self.task_table.editItem(item)
        logger.info(f"Editing task '{item.text()}'")

    def update_task(self, row: int, name: Optional[str] = None, priority: Optional[str] = None, status: Optional[str] = None):
        """Updates an existing task with new details."""
        if name:
            self.task_table.setItem(row, 0, QTableWidgetItem(name))
        if priority:
            priority_item = QTableWidgetItem(priority)
            self.apply_priority_styling(priority_item, priority)
            self.task_table.setItem(row, 1, priority_item)
        if status:
            self.task_table.setItem(row, 2, QTableWidgetItem(status))
        self.task_table.resizeColumnsToContents()

    def sort_by_priority(self):
        """Sorts the task table by the Priority column based on the PRIORITY_MAP."""
        self.task_table.sortItems(1, Qt.AscendingOrder)

    def filter_tasks(self):
        """Filters tasks by name, priority, or status."""
        query = self.search_bar.text().lower()
        for row in range(self.task_table.rowCount()):
            match = False
            for column in range(self.task_table.columnCount()):
                item = self.task_table.item(row, column)
                if item and query in item.text().lower():
                    match = True
                    break
            self.task_table.setRowHidden(row, not match)

    def open_context_menu(self, position):
        """Context menu for additional task actions."""
        menu = QMenu()

        # Mark as Completed Action
        mark_completed = QAction("Mark as Completed", self)
        mark_completed.triggered.connect(lambda: self.update_task_status(self.task_table.currentRow(), "Completed"))
        menu.addAction(mark_completed)

        # Generate Insight via Ollama Action
        generate_insight = QAction("Generate Insight via Ollama", self)
        generate_insight.triggered.connect(lambda: self.generate_insight(self.task_table.currentRow()))
        menu.addAction(generate_insight)

        # Delete Task Action
        delete_task_action = QAction("Delete Task", self)
        delete_task_action.triggered.connect(lambda: self.delete_task(self.task_table.currentRow()))
        menu.addAction(delete_task_action)

        menu.exec_(self.task_table.viewport().mapToGlobal(position))

    def update_task_status(self, row: int, status: str):
        """Updates the status of a task."""
        if row >= 0:
            self.task_table.setItem(row, 2, QTableWidgetItem(status))
            logger.info(f"Task status updated to '{status}' for row {row}")
            QMessageBox.information(self, "Task Updated", f"Task status updated to '{status}'.")

    def delete_task(self, row: int):
        """Deletes a task from the task table."""
        if row >= 0:
            task_name = self.task_table.item(row, 0).text()
            self.task_table.removeRow(row)
            logger.info(f"Task '{task_name}' at row {row} deleted.")
            QMessageBox.information(self, "Task Deleted", f"Task '{task_name}' has been deleted.")

    def generate_insight_for_price_action(self):
        """Generates an insight for TSLA price action using Ollama."""
        prompt = "Generate insight on the recent TSLA price trend."
        insight = self.ollama.run_query(prompt)
        if insight:
            QMessageBox.information(self, "Ollama Insight", f"Generated Insight: {insight}")
            self.add_insight_to_db(insight)

    def generate_insight(self, row):
        """Generates an insight for a specific task using Ollama."""
        if row >= 0:
            task_name = self.task_table.item(row, 0).text()
            prompt = f"Provide an insightful analysis for the task: {task_name}"
            response = self.ollama.run_query(prompt)
            
            # Only log the insight if it's not an error message
            if response and not response.lower().startswith("error"):
                # Add insight to the database
                insight = Insight(content=response, project_id=1)  # Assuming project_id=1 for simplicity
                self.session.add(insight)
                self.session.commit()
                # Add insight to the table
                self.add_task(response, "Low", "Insight Generated")
                QMessageBox.information(self, "Insight Generated", f"Insight: {response}")
            else:
                logger.warning(f"Failed to generate valid insight: {response}")
                QMessageBox.warning(self, "Ollama Error", f"Failed to generate insight: {response}")

    def add_insight_to_db(self, content: str):
        """Adds an insight to the database."""
        try:
            insight = Insight(content=content, project_id=1)  # Assuming project_id=1
            self.session.add(insight)
            self.session.commit()
            logger.info(f"New insight added to database: {content}")
        except Exception as e:
            logger.error(f"Error adding insight to database: {e}")

    def add_task_dialog(self):
        """Opens a dialog to add a new task."""
        task_name, ok = QInputDialog.getText(self, "Add Task", "Task Name:")
        if ok and task_name:
            priority, ok = QInputDialog.getItem(self, "Select Priority", "Priority:", ["High", "Medium", "Low"], 1, False)
            if ok and priority:
                self.add_task(task_name, priority, "Pending")
                # Add to database
                insight = Insight(content=task_name, project_id=1)  # Assuming project_id=1
                self.session.add(insight)
                self.session.commit()
                logger.info(f"New task added: {task_name} with priority {priority}")

    def show_price_chart(self):
        """Displays a matplotlib chart of TSLA's recent prices."""
        data = self.alpaca.fetch_stock_data("TSLA")
        if data:
            plt.figure(figsize=(10, 5))
            plt.plot(data['time'], data['close'], marker='o', linestyle='-')
            plt.title("TSLA Close Price Over Time")
            plt.xlabel("Date")
            plt.ylabel("Close Price ($)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            QMessageBox.warning(self, "Data Error", "Failed to fetch TSLA stock data for chart.")

    def view_alpaca_account_info(self):
        """Displays Alpaca account information."""
        account_info = self.alpaca.fetch_account_info()
        if account_info:
            info_str = (
                f"Status: {account_info['status']}\n"
                f"Cash: ${account_info['cash']}\n"
                f"Portfolio Value: ${account_info['portfolio_value']}\n"
                f"Last Equity: ${account_info['last_equity']}"
            )
            QMessageBox.information(self, "Alpaca Account Info", info_str)
        else:
            QMessageBox.warning(self, "Alpaca Error", "Failed to fetch Alpaca account information.")

    def fetch_alpaca_stock_data(self):
        """Fetches and displays TSLA stock data from Alpaca."""
        stock_data = self.alpaca.fetch_stock_data("TSLA")
        if stock_data:
            data_str = (
                f"Symbol: {stock_data['symbol']}\n"
                f"Open: ${stock_data['open']}\n"
                f"High: ${stock_data['high']}\n"
                f"Low: ${stock_data['low']}\n"
                f"Close: ${stock_data['close']}\n"
                f"Volume: {stock_data['volume']}\n"
                f"Time: {stock_data['time']}"
            )
            QMessageBox.information(self, "Alpaca TSLA Stock Data", data_str)
        else:
            QMessageBox.warning(self, "Alpaca Error", "Failed to fetch TSLA stock data from Alpaca.")
