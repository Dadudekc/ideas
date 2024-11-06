# Filename: analytics_model.py
# Description: Handles analytics data and provides summarized metrics for the dashboard.

import datetime

class AnalyticsModel:
    def __init__(self):
        self.tasks = []  # Placeholder for tasks list or data source

    def get_summary(self):
        """
        Returns a summary of project metrics, such as total tasks completed and time spent.
        """
        completed_tasks = len([task for task in self.tasks if task['status'] == 'completed'])
        return f"Tasks Completed: {completed_tasks}\nTotal Time Spent: {self.get_total_time_spent()} hours"

    def get_total_time_spent(self):
        """
        Calculates total time spent on tasks.
        """
        total_time = sum(task.get('time_spent', 0) for task in self.tasks)
        return total_time

    def get_detailed_metrics(self):
        """
        Returns a detailed report for the analytics view, including average time per task, etc.
        """
        task_count = len(self.tasks)
        avg_time = self.get_total_time_spent() / task_count if task_count > 0 else 0
        return f"Total Tasks: {task_count}\nAverage Time Per Task: {avg_time:.2f} hours"
