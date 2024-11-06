# Filename: milestone_model.py
# Description: Manages individual project milestones with enhanced tracking, analytics, and persistence.

from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from db.base import Base
import json
import logging

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MilestoneModel:
    def __init__(self, title: str, deadline: datetime, description: str = "", priority: int = 1):
        """
        Initialize a milestone with title, deadline, description, and priority level.

        Args:
            title (str): Title of the milestone.
            deadline (datetime): Deadline for the milestone.
            description (str): Description of the milestone.
            priority (int): Priority level of the milestone (1=highest).
        """
        self.title = title
        self.deadline = deadline
        self.description = description
        self.priority = priority
        self.completed = False
        self.date_completed = None
        logging.info(f"Milestone '{self.title}' initialized with priority {self.priority} and deadline {self.deadline}")

    def mark_as_completed(self):
        """Marks the milestone as completed and logs the completion date."""
        self.completed = True
        self.date_completed = datetime.now()
        logging.info(f"Milestone '{self.title}' marked as completed on {self.date_completed}")

    def is_overdue(self):
        """Checks if the milestone is overdue based on the current date."""
        overdue = not self.completed and datetime.now() > self.deadline
        if overdue:
            logging.warning(f"Milestone '{self.title}' is overdue.")
        return overdue

    def extend_deadline(self, additional_days: int):
        """Extends the milestone deadline by a specified number of days."""
        self.deadline += timedelta(days=additional_days)
        logging.info(f"Extended deadline for milestone '{self.title}' by {additional_days} days to {self.deadline}")

    def set_priority(self, new_priority: int):
        """Updates the priority of the milestone."""
        self.priority = new_priority
        logging.info(f"Updated priority for milestone '{self.title}' to {self.priority}")

    def get_summary(self):
        """Returns a summary of the milestone status."""
        status = "Completed" if self.completed else "Pending"
        overdue = " (Overdue)" if self.is_overdue() else ""
        return f"{self.title} - Priority: {self.priority} - {status}{overdue}"

    def export_to_json(self) -> str:
        """Exports the milestone details to JSON format."""
        milestone_data = {
            "title": self.title,
            "description": self.description,
            "deadline": self.deadline.isoformat(),
            "priority": self.priority,
            "completed": self.completed,
            "date_completed": self.date_completed.isoformat() if self.date_completed else None
        }
        json_data = json.dumps(milestone_data, indent=4)
        logging.info(f"Milestone '{self.title}' exported to JSON.")
        return json_data

    def __repr__(self):
        """Provides a string representation of the milestone."""
        return f"<Milestone(title={self.title}, completed={self.completed}, priority={self.priority}, deadline={self.deadline})>"

# SQLAlchemy ORM Model
class Milestone(Base):
    __tablename__ = 'milestones'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    deadline = Column(DateTime)
    priority = Column(Integer, default=1)
    completed = Column(Boolean, default=False)
    date_completed = Column(DateTime, nullable=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    
    project = relationship("Project", back_populates="milestones")
    tasks = relationship("Task", back_populates="milestone")

    def mark_as_completed(self):
        """Marks the milestone as completed and logs the completion date."""
        self.completed = True
        self.date_completed = datetime.now()
        logging.info(f"Milestone '{self.name}' marked as completed in database on {self.date_completed}")

    def is_overdue(self) -> bool:
        """Checks if the milestone is overdue based on the current date."""
        overdue = not self.completed and datetime.now() > self.deadline
        if overdue:
            logging.warning(f"Milestone '{self.name}' in database is overdue.")
        return overdue

    def __repr__(self):
        """Provides a string representation of the milestone ORM object."""
        return f"<Milestone(name='{self.name}', priority={self.priority}, project_id={self.project_id})>"

# Create a new milestone with a title, deadline, and description
milestone = MilestoneModel(
    title="Design Database Schema",
    deadline=datetime.now() + timedelta(days=7),  # 7 days from now
    description="Complete the design of the database schema for the project.",
    priority=1  # Highest priority
)

# Print initial details of the milestone
print(milestone.get_summary())  # Output includes title, priority, and status

# Check if the milestone is overdue
if milestone.is_overdue():
    print("Milestone is overdue.")
else:
    print("Milestone is on track.")

# Mark the milestone as completed
milestone.mark_as_completed()
print(milestone.get_summary())  # Should now indicate completion

# Extend the milestone deadline by 3 days (in case you want to revert completion for testing)
milestone.completed = False
milestone.extend_deadline(3)
print(milestone.get_summary())  # Output with new deadline and reset completion status

# Update priority
milestone.set_priority(2)
print(f"Updated priority: {milestone.priority}")

# Export milestone to JSON
milestone_json = milestone.export_to_json()
print("Milestone in JSON format:")
print(milestone_json)
