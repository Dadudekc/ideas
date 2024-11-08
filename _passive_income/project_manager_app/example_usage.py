# Filename: example_usage.py
# Description: Demonstrates the usage of the AIAgent class within the project management application.

from datetime import datetime, timedelta
from models.ai_agent_model import AIAgent
from models.project_model import Project
from models.milestone_model import Milestone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.base import Base
from utils.memory_manager import MemoryManager
from utils.performance_monitor import PerformanceMonitor
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Database connection string
DATABASE_URL = "postgresql+psycopg://postgres:password@localhost:5434/project_manager_db"

try:
    # Initialize the database engine and session
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    logger.info("Database engine and session created successfully.")

    # Create all tables (if not already created)
    Base.metadata.create_all(engine)
    logger.info("All tables created successfully (if not already existing).")

    # Check if a project with ID 1 exists; if not, add a default project
    project = session.query(Project).get(1)
    if not project:
        project = Project(name="Default Project", description="A default project for testing purposes.")
        session.add(project)
        session.commit()
        logger.info("Default project added to the database.")

    # Initialize memory_manager and performance_monitor
    memory_manager = MemoryManager()
    performance_monitor = PerformanceMonitor()

    # Initialize the AI Agent
    ai_agent = AIAgent(
        name="DebuggingAI",
        description="Handles debugging tasks and system diagnostics.",
        project_id=project.id,  # Use the project ID of the created project
        memory_manager=memory_manager,
        performance_monitor=performance_monitor
    )

    # Add the AI Agent to the session and commit
    session.add(ai_agent)
    session.commit()
    logger.info("AI Agent added and committed to the session.")

    # Perform tasks
    response = ai_agent.solve_task("analyze_error", error="NullReferenceException", context={"module": "auth"})
    print(f"Analyze Error Response: {response}")

    diagnostics = ai_agent.solve_task("run_diagnostics", system_check=True, detailed=True)
    print(f"Diagnostics Response: {diagnostics}")

    capabilities = ai_agent.solve_task("describe_capabilities")
    print(f"Capabilities: {capabilities}")

    # Export AI Agent details to JSON
    agent_json = ai_agent.export_to_json()
    print(f"AIAgent JSON:\n{agent_json}")

    # Shutdown the AI Agent
    ai_agent.shutdown()

except Exception as e:
    logger.error(f"An error occurred: {e}")
finally:
    # Close the session
    session.close()
    logger.info("Database session closed.")
