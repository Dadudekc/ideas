# Filename: milestone_model.py
# Description: Manages individual project milestones.

from datetime import datetime

class MilestoneModel:
    def __init__(self, title: str, deadline: datetime, description: str = ""):
        """
        Initialize a milestone with title, deadline, and optional description.
        
        Args:
            title (str): Title of the milestone.
            deadline (datetime): Deadline for the milestone.
            description (str): Description of the milestone.
        """
        self.title = title
        self.deadline = deadline
        self.description = description
        self.completed = False
        self.date_completed = None

    def mark_as_completed(self):
        """Marks the milestone as completed and logs the completion date."""
        self.completed = True
        self.date_completed = datetime.now()

    def is_overdue(self):
        """Checks if the milestone is overdue based on the current date."""
        return not self.completed and datetime.now() > self.deadline

    def get_summary(self):
        """Returns a summary of the milestone status."""
        status = "Completed" if self.completed else "Pending"
        overdue = " (Overdue)" if self.is_overdue() else ""
        return f"{self.title} - {status}{overdue}"

    def __repr__(self):
        """Provides a string representation of the milestone."""
        return f"<Milestone(title={self.title}, completed={self.completed}, deadline={self.deadline})>"
