# Filename: project_model.py
# Description: Manages overall project details, milestones, and progress tracking.

from typing import List
from datetime import datetime
from models.milestone_model import MilestoneModel

class ProjectModel:
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a project with a name, description, and an empty list of milestones.
        
        Args:
            name (str): Name of the project.
            description (str): Description of the project.
        """
        self.name = name
        self.description = description
        self.milestones: List[MilestoneModel] = []

    def add_milestone(self, milestone: MilestoneModel):
        """Adds a milestone to the project."""
        self.milestones.append(milestone)

    def get_progress(self) -> float:
        """Calculates the percentage of completed milestones."""
        if not self.milestones:
            return 0.0
        completed = sum(1 for milestone in self.milestones if milestone.completed)
        return (completed / len(self.milestones)) * 100

    def get_status_summary(self) -> str:
        """Provides a summary of the project status, including milestone details."""
        status_lines = [f"Project: {self.name}", f"Description: {self.description}", f"Progress: {self.get_progress():.2f}%"]
        status_lines.extend(milestone.get_summary() for milestone in self.milestones)
        return "\n".join(status_lines)

    def get_overdue_milestones(self) -> List[MilestoneModel]:
        """Returns a list of overdue milestones."""
        return [milestone for milestone in self.milestones if milestone.is_overdue()]

    def __repr__(self):
        """Provides a string representation of the project."""
        return f"<Project(name={self.name}, milestones={len(self.milestones)})>"
