# Filename: project_model.py
# Description: Manages project details, milestones, and progress tracking, with advanced tracking and analytics.

from typing import List, Optional
from datetime import datetime, timedelta
from models.milestone_model import MilestoneModel
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from db.base import Base
import json
import logging

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ProjectModel:
    def __init__(self, name: str, description: str = "", start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        self.name = name
        self.description = description
        self.start_date = start_date or datetime.now()
        self.end_date = end_date
        self.milestones: List[MilestoneModel] = []
        logging.info(f"Project '{self.name}' initialized with start date: {self.start_date}")

    def add_milestone(self, milestone: MilestoneModel):
        """Adds a milestone to the project."""
        self.milestones.append(milestone)
        logging.info(f"Milestone '{milestone.title}' added to project '{self.name}'.")

    def reset_milestones(self):
        """Resets all milestones to an incomplete state."""
        for milestone in self.milestones:
            milestone.completed = False
        logging.info(f"All milestones in project '{self.name}' have been reset.")

    def get_progress(self) -> float:
        """Calculates the percentage of completed milestones."""
        if not self.milestones:
            logging.warning(f"No milestones in project '{self.name}' to calculate progress.")
            return 0.0
        completed = sum(1 for milestone in self.milestones if milestone.completed)
        progress = (completed / len(self.milestones)) * 100
        logging.info(f"Progress for project '{self.name}': {progress:.2f}%")
        return progress

    def get_milestones_by_status(self, completed: bool = True) -> List[MilestoneModel]:
        """Fetches milestones by their completion status."""
        status_milestones = [milestone for milestone in self.milestones if milestone.completed == completed]
        logging.info(f"Retrieved {len(status_milestones)} {'completed' if completed else 'incomplete'} milestones.")
        return status_milestones

    def get_status_summary(self) -> str:
        """Provides a summary of the project status."""
        status_lines = [
            f"Project: {self.name}",
            f"Description: {self.description}",
            f"Start Date: {self.start_date.strftime('%Y-%m-%d')}",
            f"End Date: {self.end_date.strftime('%Y-%m-%d') if self.end_date else 'Not specified'}",
            f"Progress: {self.get_progress():.2f}%"
        ]
        status_lines.extend(milestone.get_summary() for milestone in self.milestones)
        summary = "\n".join(status_lines)
        logging.info(f"Status summary for project '{self.name}' generated.")
        return summary

    def get_overdue_milestones(self) -> List[MilestoneModel]:
        """Returns a list of overdue milestones."""
        overdue_milestones = [milestone for milestone in self.milestones if milestone.is_overdue()]
        logging.info(f"Found {len(overdue_milestones)} overdue milestones in project '{self.name}'.")
        return overdue_milestones

    def calculate_remaining_time(self) -> Optional[float]:
        """Calculates the remaining time in days until the project end date."""
        if not self.end_date:
            logging.warning(f"End date not set for project '{self.name}'. Cannot calculate remaining time.")
            return None
        remaining_days = (self.end_date - datetime.now()).days
        logging.info(f"Remaining time for project '{self.name}': {remaining_days} days")
        return max(0, remaining_days)

    def get_milestone_by_name(self, milestone_name: str) -> Optional[MilestoneModel]:
        """Fetches a milestone by name."""
        for milestone in self.milestones:
            if milestone.title == milestone_name:
                logging.info(f"Milestone '{milestone_name}' found in project '{self.name}'.")
                return milestone
        logging.warning(f"Milestone '{milestone_name}' not found in project '{self.name}'.")
        return None

    def remove_milestone(self, milestone_name: str) -> bool:
        """Removes a milestone by name and returns success status."""
        milestone = self.get_milestone_by_name(milestone_name)
        if milestone:
            self.milestones.remove(milestone)
            logging.info(f"Milestone '{milestone_name}' removed from project '{self.name}'.")
            return True
        logging.warning(f"Attempted to remove non-existing milestone '{milestone_name}' from project '{self.name}'.")
        return False

    def extend_end_date(self, additional_days: int):
        """Extends the project end date by a specified number of days."""
        if self.end_date:
            self.end_date += timedelta(days=additional_days)
            logging.info(f"Extended end date for project '{self.name}' by {additional_days} days.")
        else:
            logging.warning(f"End date not set for project '{self.name}'. Cannot extend end date.")

    def export_to_json(self) -> str:
        """Exports the project details and milestones to JSON."""
        project_data = {
            "name": self.name,
            "description": self.description,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "milestones": [milestone.to_dict() for milestone in self.milestones]
        }
        json_data = json.dumps(project_data, indent=4)
        logging.info(f"Exported project '{self.name}' to JSON format.")
        return json_data

    def __repr__(self):
        """Provides a string representation of the project with details."""
        return f"<Project(name={self.name}, milestones={len(self.milestones)}, progress={self.get_progress():.2f}%)>"

# SQLAlchemy ORM Model
class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    start_date = Column(DateTime, default=datetime.now)
    end_date = Column(DateTime)
    
    milestones = relationship("Milestone", back_populates="project")
    tasks = relationship("Task", back_populates="project")
    ai_agents = relationship("AIAgent", back_populates="project")

    def __repr__(self):
        return f"<Project(name='{self.name}')>"

# Example usage
if __name__ == "__main__":
    # Initialize a new project
    project = ProjectModel(
        name="AI SaaS Platform",
        description="Develop an AI-driven SaaS platform with project management features.",
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=90)  # Project deadline in 90 days
    )

    # Step 1: Add milestones
    milestone1 = MilestoneModel(
        title="Design UI",
        deadline=datetime.now() + timedelta(days=14),
        description="Complete the design of the user interface.",
        priority=2
    )

    milestone2 = MilestoneModel(
        title="Build API",
        deadline=datetime.now() + timedelta(days=30),
        description="Develop the RESTful API backend.",
        priority=1
    )

    # Add milestones to the project
    project.add_milestone(milestone1)
    project.add_milestone(milestone2)

    # Step 2: View project summary
    print("Project Status Summary:")
    print(project.get_status_summary())

    # Step 3: Calculate project progress
    progress = project.get_progress()
    print(f"Project Progress: {progress:.2f}%")

    # Step 4: Check for overdue milestones
    overdue_milestones = project.get_overdue_milestones()
    print("Overdue Milestones:")
    for milestone in overdue_milestones:
        print(milestone.get_summary())

    # Step 5: Mark a milestone as completed
    milestone1.mark_as_completed()
    print("Updated Project Status Summary after completing a milestone:")
    print(project.get_status_summary())

    # Step 6: Extend the project end date by 15 days
    project.extend_end_date(15)
    print(f"New project end date: {project.end_date}")

    # Step 7: Export project details to JSON
    project_json = project.export_to_json()
    print("Project in JSON format:")
    print(project_json)
