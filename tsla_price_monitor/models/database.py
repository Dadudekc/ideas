# models/database.py
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Boolean,
    Float,
    create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from typing import Optional, Dict, Any
from utils.logger import logger
from config import Config

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, default="")
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)

    milestones = relationship("Milestone", back_populates="project", cascade="all, delete-orphan")
    insights = relationship("Insight", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project(name='{self.name}', milestones={len(self.milestones)}, insights={len(self.insights)})>"

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
    insights = relationship("Insight", back_populates="milestone", cascade="all, delete-orphan")

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

class Insight(Base):
    __tablename__ = 'insights'
    id = Column(Integer, primary_key=True)
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    milestone_id = Column(Integer, ForeignKey('milestones.id'), nullable=True)

    project = relationship("Project", back_populates="insights")
    milestone = relationship("Milestone", back_populates="insights")

    def __init__(self, content: str, project_id: int, milestone_id: Optional[int] = None):
        self.content = content
        self.project_id = project_id
        self.milestone_id = milestone_id
        logger.info(f"Insight created: {self.content}")

    def export_to_json(self) -> Dict[str, Any]:
        """Exports the insight details to a dictionary."""
        insight_data = {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "project_id": self.project_id,
            "milestone_id": self.milestone_id,
        }
        logger.info(f"Insight '{self.content}' exported to JSON format.")
        return insight_data

    def __repr__(self) -> str:
        """Provides a string representation of the insight ORM object."""
        return f"<Insight(content='{self.content}', timestamp={self.timestamp})>"

def get_engine():
    """Creates and returns the SQLAlchemy engine."""
    return create_engine(Config.DATABASE_URL, echo=False, future=True)

def get_session():
    """Creates and returns a new SQLAlchemy session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return Session()

def init_db():
    """Initializes the database and creates tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database initialized and tables created.")
