# Filename: utils/memory_manager.py
# Description: Manages memory storage and retrieval for AI agents.

import logging
from typing import Optional, List

# Configure logging for the MemoryManager
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MemoryManager:
    def __init__(self):
        # Initialize memory storage (could be a database, in-memory store, etc.)
        self.memory_store: Dict[str, List[str]] = {}
        logger.info("MemoryManager initialized.")

    def retrieve_memory(self, agent_name: str, limit: int = 5) -> str:
        """Retrieves the last 'limit' memory entries for the specified agent."""
        memories = self.memory_store.get(agent_name, [])
        retrieved = "\n".join(memories[-limit:])
        logger.debug(f"Retrieved memory for agent '{agent_name}': {retrieved}")
        return retrieved

    def save_memory(self, agent_name: str, prompt: str, response: str):
        """Saves a new memory entry for the specified agent."""
        if agent_name not in self.memory_store:
            self.memory_store[agent_name] = []
        memory_entry = f"User: {prompt}\nAI: {response}"
        self.memory_store[agent_name].append(memory_entry)
        logger.debug(f"Saved memory for agent '{agent_name}': {memory_entry}")
