# Filename: task_model.py
# Manages task data and status for each project task.

from typing import List, Dict

class Task:
    def __init__(self, title: str, status: str = "Pending"):
        self.title = title
        self.status = status

    def update_status(self, new_status: str):
        self.status = new_status

class TaskManager:
    def __init__(self):
        self.tasks: List[Task] = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def get_tasks(self) -> List[Dict[str, str]]:
        return [{"title": task.title, "status": task.status} for task in self.tasks]
