# Filename: utils/performance_monitor.py
# Description: Monitors and logs performance metrics for AI agents.

import logging
from typing import Optional, Dict, Any

# Configure logging for the PerformanceMonitor
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PerformanceMonitor:
    def __init__(self):
        # Initialize performance tracking (could be a database, logs, etc.)
        self.performance_data: Dict[str, Dict[str, Any]] = {}
        logger.info("PerformanceMonitor initialized.")

    def log_performance(self, agent_name: str, prompt: str, success: bool, response: str):
        """Logs the performance of a given task."""
        if agent_name not in self.performance_data:
            self.performance_data[agent_name] = {"successes": 0, "failures": 0, "failure_details": []}
        if success:
            self.performance_data[agent_name]["successes"] += 1
            logger.debug(f"Logged success for agent '{agent_name}'. Total successes: {self.performance_data[agent_name]['successes']}")
        else:
            self.performance_data[agent_name]["failures"] += 1
            self.performance_data[agent_name]["failure_details"].append(response)
            logger.debug(f"Logged failure for agent '{agent_name}'. Total failures: {self.performance_data[agent_name]['failures']}")

    def analyze_performance(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Analyzes the performance data of a given agent."""
        data = self.performance_data.get(agent_name)
        if not data:
            logger.warning(f"No performance data available for agent '{agent_name}'.")
            return None
        total = data["successes"] + data["failures"]
        success_rate = (data["successes"] / total) * 100 if total > 0 else 100
        analysis = {
            "success_rate": success_rate,
            "failures": data["failures"],
            "failure_details": data["failure_details"]
        }
        logger.info(f"Performance analysis for agent '{agent_name}': {analysis}")
        return analysis
