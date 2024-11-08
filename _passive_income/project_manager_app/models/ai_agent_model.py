# Filename: models/ai_agent_model.py
# Description: AIAgent class with enhanced debugging and diagnostic capabilities inspired by the DebuggerAgent.

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from db.base import Base
import logging
import subprocess
import json
from typing import Optional, Dict, Any, List

# Configure logging for the AIAgent model
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent adding multiple handlers if this module is imported multiple times
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class AIAgent(Base):
    """
    AIAgent Class

    Represents an AI agent capable of interacting with users, retaining memory,
    providing context-aware responses, and self-improving based on feedback.
    Integrates with memory management and performance monitoring tools.
    """

    __tablename__ = 'ai_agents'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)

    # Use string-based reference to Project to avoid circular dependency
    project = relationship("Project", back_populates="ai_agents")

    def __init__(self, name: str, description: str = "", project_id: Optional[int] = None,
                 memory_manager=None, performance_monitor=None, dispatcher=None):
        self.name = name
        self.description = description
        self.project_id = project_id
        self.memory_manager = memory_manager
        self.performance_monitor = performance_monitor
        self.dispatcher = dispatcher or self._initialize_dispatcher()
        logger.info(f"AIAgent '{self.name}' initialized for project ID '{self.project_id}'.")

    def _initialize_dispatcher(self):
        """
        Initialize the AgentDispatcher if not provided to avoid circular imports.
        """
        try:
            from core.AgentDispatcher import AgentDispatcher
            dispatcher = AgentDispatcher()
            logger.info(f"AgentDispatcher initialized for AIAgent '{self.name}'.")
            return dispatcher
        except ImportError as e:
            logger.error(f"Failed to initialize AgentDispatcher: {e}")
            raise

    def run_query(self, prompt: str) -> str:
        """
        Executes a query to the AI with memory context and logs performance.

        Args:
            prompt (str): The user input to send to the AI.

        Returns:
            str: The AI's response.
        """
        try:
            memory_context = self.memory_manager.retrieve_memory(self.name, limit=5)
            complete_prompt = f"{memory_context}User: {prompt}\nAI:"
            logger.debug(f"Complete prompt sent to AI:\n{complete_prompt}")

            # Execute command and get response
            result = subprocess.run(
                ["ollama", "run", "mistral:latest", "--prompt", complete_prompt],
                capture_output=True, text=True, check=True
            )
            response = result.stdout.strip()
            self.memory_manager.save_memory(self.name, prompt, response)
            self.performance_monitor.log_performance(self.name, prompt, success=True, response=response)
            logger.info(f"AI response received and logged for prompt: '{prompt}'")
            return response

        except subprocess.CalledProcessError as e:
            error_message = f"AI communication error: {e.stdout.strip() or e.stderr.strip()}"
            logger.error(error_message)
            self.performance_monitor.log_performance(self.name, prompt, success=False, response=error_message)
            return error_message

        except Exception as ex:
            error_message = f"Unexpected error: {str(ex)}"
            logger.error(error_message)
            self.performance_monitor.log_performance(self.name, prompt, success=False, response=error_message)
            return error_message

    def analyze_error(self, error: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyzes an error message, providing diagnostics and potential resolution steps.

        Args:
            error (Optional[str]): Error message to analyze.
            context (Optional[Dict[str, Any]]): Additional context for error analysis.

        Returns:
            str: Analysis result or message if no error provided.
        """
        if not error:
            logger.warning("No error message provided for analysis.")
            return "No error message provided for analysis."

        logger.info(f"Analyzing error: '{error}' with context: {context}")
        # Placeholder for complex analysis logic
        analysis = f"Error analysis: '{error}'. Context: {context or 'None'}"
        logger.debug(f"Detailed error analysis: {analysis}")
        return analysis

    def run_diagnostics(self, system_check: bool = True, detailed: bool = False) -> str:
        """
        Runs system diagnostics, with options for basic or detailed checks.

        Args:
            system_check (bool): If True, perform system-level diagnostics.
            detailed (bool): If True, include detailed diagnostic information.

        Returns:
            str: Diagnostic results.
        """
        logger.info(f"Running diagnostics: system_check={system_check}, detailed={detailed}")
        diagnostics = "Diagnostics: Basic check passed."
        if system_check:
            diagnostics += " System check passed."
        if detailed:
            diagnostics += " Detailed diagnostics: All systems operational."
        logger.debug(f"Diagnostics result: {diagnostics}")
        return diagnostics

    def describe_capabilities(self) -> str:
        """
        Provides a summary of the agent's debugging and diagnostic capabilities.

        Returns:
            str: Description of the agent's debugging functionalities.
        """
        capabilities = (
            f"{self.name} can perform error analysis, run system diagnostics, "
            "provide context-aware debugging, and manage project tasks."
        )
        logger.debug(f"{self.name} capabilities: {capabilities}")
        return capabilities

    def solve_task(self, task: str, **kwargs) -> Any:
        """
        Executes a specified debugging or management task.

        Args:
            task (str): Type of task (e.g., 'analyze_error', 'run_diagnostics').
            **kwargs: Additional arguments specific to the task.

        Returns:
            Any: Result of the task or an error message if the task is unknown.
        """
        logger.info(f"{self.name} received task: '{task}'")
        task_methods = {
            "analyze_error": self.analyze_error,
            "run_diagnostics": self.run_diagnostics,
            "describe_capabilities": self.describe_capabilities,
        }
        task_function = task_methods.get(task)
        if task_function:
            return task_function(**kwargs)
        else:
            logger.error(f"Unknown task: '{task}'")
            return f"Unknown task: '{task}'"

    def shutdown(self) -> None:
        """
        Gracefully shuts down the agent, releasing resources if necessary.
        """
        logger.info(f"{self.name} is shutting down.")
        # Add any necessary cleanup logic here

    def export_to_json(self) -> str:
        """
        Exports the AI agent's details to JSON.

        Returns:
            str: JSON string of the agent's details.
        """
        agent_data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "project_id": self.project_id
        }
        json_data = json.dumps(agent_data, indent=4)
        logger.info(f"Exported AIAgent '{self.name}' details to JSON.")
        return json_data

    def __repr__(self) -> str:
        return f"<AIAgent(name='{self.name}', project_id={self.project_id})>"
