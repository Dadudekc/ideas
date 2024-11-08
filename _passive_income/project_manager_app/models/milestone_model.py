# Filename: milestone_model.py
# Description: Manages individual project milestones with enhanced tracking, analytics, and persistence.

import sys
import os
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Dynamically set the root directory for imports and check if it's applied correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added project root to sys.path: {project_root}")
else:
    print(f"Project root already in sys.path: {project_root}")

# Print sys.path for debugging
print("Current sys.path:")
for path in sys.path:
    print(f" - {path}")

try:
    from db.base import Base  # Import after setting sys.path
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure that 'db.base' exists at the expected path relative to 'milestone_model.py'.")
    raise

# Configure logging for the Milestone model
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Milestone(Base):
    __tablename__ = 'milestones'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    deadline = Column(DateTime, nullable=True)
    priority = Column(Integer, default=1)
    completed = Column(Boolean, default=False)
    date_completed = Column(DateTime, nullable=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    progress = Column(Float, default=0.0)  # Percentage of completion

    project = relationship("Project", back_populates="milestones")
    tasks = relationship("Task", back_populates="milestone", cascade="all, delete-orphan")

    def __init__(self, name: str, description: str = "", deadline: Optional[datetime] = None, priority: int = 1, project_id: Optional[int] = None):
        self.name = name
        self.description = description
        self.deadline = deadline or (datetime.utcnow() + timedelta(days=7))  # Default deadline set 7 days from now
        self.priority = priority
        self.project_id = project_id
        logger.info(f"Milestone '{self.name}' initialized with priority {self.priority} and deadline {self.deadline}")

    def mark_as_completed(self):
        """Marks the milestone as completed and logs the completion date."""
        self.completed = True
        self.date_completed = datetime.utcnow()
        self.progress = 100.0
        logger.info(f"Milestone '{self.name}' marked as completed on {self.date_completed}")

    def is_overdue(self) -> bool:
        """Checks if the milestone is overdue based on the current date."""
        overdue = not self.completed and datetime.utcnow() > self.deadline
        if overdue:
            logger.warning(f"Milestone '{self.name}' is overdue.")
        return overdue

    def extend_deadline(self, additional_days: int):
        """Extends the milestone deadline by a specified number of days."""
        self.deadline += timedelta(days=additional_days)
        logger.info(f"Extended deadline for milestone '{self.name}' by {additional_days} days to {self.deadline}")

    def set_priority(self, new_priority: int):
        """Updates the priority of the milestone."""
        self.priority = new_priority
        logger.info(f"Updated priority for milestone '{self.name}' to {self.priority}")

    def update_progress(self, completion_rate: float):
        """Updates progress and adjusts estimated completion based on progress rate."""
        self.progress = min(max(completion_rate, 0.0), 100.0)  # Clamp between 0 and 100%
        logger.info(f"Milestone '{self.name}' progress updated to {self.progress}%")

        if self.progress >= 100.0:
            self.mark_as_completed()

    def estimated_completion(self) -> Optional[datetime]:
        """Calculates an estimated completion date based on progress and priority."""
        if self.completed or self.progress <= 0:
            return None
        days_passed = (datetime.utcnow() - (self.deadline - timedelta(days=7))).days
        days_needed = days_passed / (self.progress / 100) if self.progress > 0 else 0
        estimated_end = datetime.utcnow() + timedelta(days=days_needed)
        logger.info(f"Estimated completion for milestone '{self.name}': {estimated_end}")
        return estimated_end

    def dynamic_deadline_adjustment(self):
        """Adjusts the deadline dynamically based on priority and progress to keep milestones on track."""
        if self.progress < 50 and self.priority == 1:
            self.extend_deadline(3)  # High-priority tasks gain a short extension
        elif self.progress < 50 and self.priority > 1:
            self.extend_deadline(7)  # Lower-priority tasks gain a more generous extension

    def get_summary(self) -> str:
        """Returns a summary of the milestone status."""
        status = "Completed" if self.completed else "Pending"
        overdue = " (Overdue)" if self.is_overdue() else ""
        progress_display = f"{self.progress:.1f}%"
        summary = f"{self.name} - Priority: {self.priority} - {status}{overdue} - Progress: {progress_display}"
        logger.info(f"Summary for milestone '{self.name}': {summary}")
        return summary

    def export_to_json(self) -> Dict[str, Any]:
        """Exports the milestone details to a dictionary."""
        milestone_data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "priority": self.priority,
            "completed": self.completed,
            "date_completed": self.date_completed.isoformat() if self.date_completed else None,
            "project_id": self.project_id,
            "progress": self.progress,
            "estimated_completion": self.estimated_completion().isoformat() if self.estimated_completion() else None
        }
        logger.info(f"Milestone '{self.name}' exported to JSON format.")
        return milestone_data

    def __repr__(self) -> str:
        """Provides a string representation of the milestone ORM object."""
        return f"<Milestone(name='{self.name}', completed={self.completed}, priority={self.priority}, deadline={self.deadline}, progress={self.progress}%)>"
