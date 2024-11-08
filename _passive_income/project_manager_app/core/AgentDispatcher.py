# Filename: core/AgentDispatcher.py
# Description: Manages AI agent task delegation, dynamic agent selection, and error handling.

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
from utils.memory_manager import MemoryManager
from utils.performance_monitor import PerformanceMonitor
import json

# Add project root to the system path to resolve import errors
project_root = Path(__file__).resolve().parents[2]  # Adjust as necessary based on project structure
sys.path.append(str(project_root))

# Import AIAgent after adding the root path
from models.ai_agent_model import AIAgent

# Configure logging for the dispatcher
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class AgentDispatcher:
    """
    AgentDispatcher Class

    Handles the initialization, selection, and delegation of tasks to AI agents.
    Dynamically chooses an agent based on task type and agent capabilities.
    """

    def __init__(self):
        """Initializes the dispatcher with memory manager and performance monitor."""
        self.agents: Dict[str, AIAgent] = {}
        self.memory_manager = MemoryManager()
        self.performance_monitor = PerformanceMonitor()
        logger.info("AgentDispatcher initialized.")

    def register_agent(self, agent: AIAgent):
        """
        Registers a new agent with the dispatcher.

        Args:
            agent (AIAgent): The AI agent to register.
        """
        self.agents[agent.name] = agent
        logger.info(f"Registered agent '{agent.name}' with AgentDispatcher.")

    def get_agent(self, task_type: str) -> Optional[AIAgent]:
        """
        Retrieves the appropriate agent for a given task type.

        Args:
            task_type (str): The type of task to find an agent for.

        Returns:
            Optional[AIAgent]: The selected AI agent or None if no suitable agent is found.
        """
        for agent in self.agents.values():
            if task_type in agent.describe_capabilities():
                logger.info(f"Selected agent '{agent.name}' for task '{task_type}'.")
                return agent
        logger.warning(f"No agent found capable of handling task '{task_type}'.")
        return None

    def execute_task(self, task: str, task_type: str, **kwargs) -> Any:
        """
        Executes a specified task by delegating it to the appropriate agent.

        Args:
            task (str): The specific task to execute (e.g., 'analyze_error').
            task_type (str): The general type of task (e.g., 'debugging', 'diagnostics').
            **kwargs: Additional arguments required for task execution.

        Returns:
            Any: Result of the task execution or an error message if the task could not be executed.
        """
        agent = self.get_agent(task_type)
        if agent:
            logger.info(f"Executing task '{task}' using agent '{agent.name}' with kwargs: {kwargs}")
            return agent.solve_task(task, **kwargs)
        else:
            error_message = f"No suitable agent found for task type '{task_type}'."
            logger.error(error_message)
            return error_message

    def list_registered_agents(self) -> List[str]:
        """
        Lists all registered agents and their capabilities.

        Returns:
            List[str]: List of agent names and their capabilities.
        """
        agent_summaries = []
        for agent in self.agents.values():
            capabilities = agent.describe_capabilities()
            summary = f"Agent: {agent.name}, Capabilities: {capabilities}"
            agent_summaries.append(summary)
            logger.info(summary)
        return agent_summaries

    def analyze_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes and reports performance metrics for all registered agents.

        Returns:
            Dict[str, Dict[str, Any]]: Performance data for each agent.
        """
        performance_data = {}
        for agent in self.agents.values():
            performance = self.performance_monitor.analyze_performance(agent.name)
            performance_data[agent.name] = performance
            logger.info(f"Performance for agent '{agent.name}': {performance}")
        return performance_data

    def shutdown_all_agents(self):
        """Gracefully shuts down all registered agents."""
        for agent in self.agents.values():
            agent.shutdown()
            logger.info(f"Agent '{agent.name}' has been shut down.")

    def load_agents(self, agent_configs: List[Dict[str, Any]]):
        """
        Loads and registers agents from a list of configurations.

        Args:
            agent_configs (List[Dict[str, Any]]): List of agent configurations with keys 'name', 'description', 'project_id'.
        """
        for config in agent_configs:
            try:
                agent = AIAgent(
                    name=config['name'],
                    description=config.get('description', ''),
                    project_id=config.get('project_id'),
                    memory_manager=self.memory_manager,
                    performance_monitor=self.performance_monitor
                )
                self.register_agent(agent)
            except Exception as e:
                logger.error(f"Failed to load agent '{config['name']}': {e}")

    def suggest_improvements(self):
        """Aggregates improvement suggestions across all agents based on performance data."""
        for agent in self.agents.values():
            agent.suggest_improvements()
            logger.info(f"Improvement suggestions generated for agent '{agent.name}'.")

    def export_agents_to_json(self) -> str:
        """
        Exports all registered agents' details to JSON format.

        Returns:
            str: JSON string of all agent details.
        """
        agents_data = {agent.name: json.loads(agent.export_to_json()) for agent in self.agents.values()}
        json_data = json.dumps(agents_data, indent=4)
        logger.info("Exported all agents' details to JSON.")
        return json_data

# Example usage
if __name__ == "__main__":
    dispatcher = AgentDispatcher()

    # Example configuration list
    agent_configs = [
        {"name": "DebugAgent", "description": "Handles debugging tasks", "project_id": 1},
        {"name": "DiagAgent", "description": "Handles diagnostics tasks", "project_id": 2}
    ]

    # Load and register agents
    dispatcher.load_agents(agent_configs)

    # List registered agents
    print("Registered Agents and Capabilities:")
    for agent_summary in dispatcher.list_registered_agents():
        print(agent_summary)

    # Execute a task
    result = dispatcher.execute_task("analyze_error", task_type="debugging", error="NullReferenceException", context={"module": "auth"})
    print("Task Result:", result)

    # Analyze performance
    performance_report = dispatcher.analyze_performance()
    print("Performance Report:", performance_report)

    # Suggest improvements
    dispatcher.suggest_improvements()

    # Export agents to JSON
    json_data = dispatcher.export_agents_to_json()
    print("Agents JSON:", json_data)

    # Shutdown all agents
    dispatcher.shutdown_all_agents()
