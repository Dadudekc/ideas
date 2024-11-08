# Filename: models/task_model.py
# Description: Manages individual tasks within milestones, tracking progress and completion.

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from db.base import Base
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

# Configure logging for the Task model
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    deadline = Column(DateTime, nullable=True)
    completed = Column(Boolean, default=False)
    date_completed = Column(DateTime, nullable=True)
    milestone_id = Column(Integer, ForeignKey('milestones.id'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    
    milestone = relationship("Milestone", back_populates="tasks")
    project = relationship("Project", back_populates="tasks")
    
    def __init__(self, name: str, description: str = "", deadline: Optional[datetime] = None, milestone_id: Optional[int] = None, project_id: Optional[int] = None):
        self.name = name
        self.description = description
        self.deadline = deadline or (datetime.utcnow() + timedelta(days=3))
        self.milestone_id = milestone_id
        self.project_id = project_id
        logger.info(f"Task '{self.name}' initialized with deadline {self.deadline}")

    def mark_as_completed(self):
        """Marks the task as completed and logs the completion date."""
        self.completed = True
        self.date_completed = datetime.utcnow()
        logger.info(f"Task '{self.name}' marked as completed on {self.date_completed}")

    def is_overdue(self) -> bool:
        """Checks if the task is overdue based on the current date."""
        overdue = not self.completed and datetime.utcnow() > self.deadline
        if overdue:
            logger.warning(f"Task '{self.name}' is overdue.")
        return overdue

    def __repr__(self) -> str:
        return f"<Task(name='{self.name}', completed={self.completed}, deadline={self.deadline})>"

    def export_to_json(self) -> Dict[str, Any]:
        """Exports the task details to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "completed": self.completed,
            "date_completed": self.date_completed.isoformat() if self.date_completed else None,
            "milestone_id": self.milestone_id,
            "project_id": self.project_id
        }
