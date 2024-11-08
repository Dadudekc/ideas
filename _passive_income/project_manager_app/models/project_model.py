# Filename: project_model.py
# Description: Manages project details, milestones, and progress tracking, with advanced tracking and analytics.

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from db.base import Base
import json
import logging

# Configure logger for the Project model
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)
    
    milestones = relationship("Milestone", back_populates="project", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    ai_agents = relationship("AIAgent", back_populates="project", cascade="all, delete-orphan")

    def __init__(self, name: str, description: str = "", start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        self.name = name
        self.description = description
        self.start_date = start_date or datetime.utcnow()
        self.end_date = end_date
        logger.info(f"Project '{self.name}' initialized with start date: {self.start_date}")

    def add_milestone(self, milestone: 'Milestone'):
        """Adds a milestone to the project."""
        self.milestones.append(milestone)
        logger.info(f"Milestone '{milestone.name}' added to project '{self.name}'.")

    def get_summary(self) -> str:
        """Provides a summary of the project status."""
        progress = self.get_progress()
        summary = (
            f"Project: {self.name}\n"
            f"Description: {self.description}\n"
            f"Start Date: {self.start_date.strftime('%Y-%m-%d')}\n"
            f"End Date: {self.end_date.strftime('%Y-%m-%d') if self.end_date else 'Not specified'}\n"
            f"Progress: {progress:.2f}%\n"
            f"Milestones: {len(self.milestones)}\n"
            f"AI Agents: {len(self.ai_agents)}"
        )
        logger.info(f"Status summary for project '{self.name}' generated.")
        return summary

    def get_progress(self) -> float:
        """Calculates the percentage of completed milestones."""
        if not self.milestones:
            logger.warning(f"No milestones in project '{self.name}' to calculate progress.")
            return 0.0
        completed = sum(1 for milestone in self.milestones if milestone.completed)
        progress = (completed / len(self.milestones)) * 100
        logger.info(f"Progress for project '{self.name}': {progress:.2f}%")
        return progress

    def get_overdue_milestones(self) -> List['Milestone']:
        """Returns a list of overdue milestones."""
        overdue_milestones = [milestone for milestone in self.milestones if milestone.is_overdue()]
        logger.info(f"Found {len(overdue_milestones)} overdue milestones in project '{self.name}'.")
        return overdue_milestones

    def extend_end_date(self, additional_days: int):
        """Extends the project end date by a specified number of days."""
        if self.end_date:
            self.end_date += timedelta(days=additional_days)
            logger.info(f"Extended end date for project '{self.name}' by {additional_days} days.")
        else:
            logger.warning(f"End date not set for project '{self.name}'. Cannot extend end date.")

    def export_to_json(self) -> str:
        """Exports the project details and milestones to JSON."""
        project_data = {
            "name": self.name,
            "description": self.description,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "milestones": [milestone.to_json() for milestone in self.milestones]
        }
        json_data = json.dumps(project_data, indent=4)
        logger.info(f"Exported project '{self.name}' to JSON format.")
        return json_data

    def __repr__(self):
        return f"<Project(name='{self.name}', milestones={len(self.milestones)}, ai_agents={len(self.ai_agents)})>"
