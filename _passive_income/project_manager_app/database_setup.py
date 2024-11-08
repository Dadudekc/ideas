# Filename: database_setup.py
# Description: Sets up the database, creates tables, and optionally adds initial data for testing.

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.base import Base
import logging
import psycopg
from sqlalchemy.exc import OperationalError

# Configure logging for setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Import models to define tables
from models.project_model import Project
from models.milestone_model import Milestone
from models.task_model import Task
from models.ai_agent_model import AIAgent

# Database connection string
DATABASE_URL = "postgresql+psycopg://postgres:password@localhost:5434/project_manager_db"

def create_database_if_not_exists():
    """Checks for the database's existence and creates it if necessary."""
    try:
        # Connect to the default database to check or create `project_manager_db`
        with psycopg.connect("postgresql://postgres:password@localhost:5434/postgres") as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = 'project_manager_db'")
                if not cur.fetchone():
                    cur.execute("CREATE DATABASE project_manager_db")
                    logger.info("Database 'project_manager_db' created successfully.")
                else:
                    logger.info("Database 'project_manager_db' already exists.")
    except Exception as e:
        logger.error(f"Error checking or creating database: {e}")
        raise

# Initialize the database engine
def setup_engine_and_create_tables():
    """Sets up the database engine and creates tables."""
    try:
        engine = create_engine(DATABASE_URL)
        logger.info("Database engine created successfully.")
        
        # Create tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully.")
        
        # Return sessionmaker bound to this engine
        return sessionmaker(bind=engine)
    except OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to set up database engine or create tables: {e}")
        raise

# Function to add initial data
def add_initial_data(session):
    """Adds initial data to the database for testing purposes."""
    try:
        project = Project(name="Initial Project", description="Sample project for testing.")
        milestone = Milestone(name="Initial Milestone", description="First milestone in the project", project=project)
        task = Task(name="Initial Task", description="First task in the milestone", milestone=milestone, project=project)

        session.add_all([project, milestone, task])
        session.commit()
        logger.info("Initial data added successfully.")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to add initial data: {e}")

# Run setup steps
create_database_if_not_exists()  # Ensure database exists
Session = setup_engine_and_create_tables()  # Set up engine and sessionmaker
session = Session()

# Add initial data (Uncomment to enable)
# add_initial_data(session)

# Close the session when done
def close_session(session):
    """Closes the database session gracefully."""
    try:
        session.close()
        logger.info("Database session closed successfully.")
    except Exception as e:
        logger.error(f"Failed to close database session: {e}")

close_session(session)
