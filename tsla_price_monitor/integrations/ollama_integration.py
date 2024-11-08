# integrations/ollama_integration.py
import subprocess
from typing import Optional
from utils.logger import logger
from config import Config

class OllamaIntegration:
    """
    Handles interaction with the Ollama AI model via CLI.
    Ensure Ollama is installed and accessible via the command line.
    """
    def __init__(self, model_name: str = "default-model"):  # Replace "default-model" with the actual model name if known
        self.command = Config.OLLAMA_COMMAND
        self.model_name = model_name

    def run_query(self, prompt: str) -> str:
        """Runs a query using Ollama and returns the response."""
        try:
            # Update command format without the --prompt flag
            result = subprocess.run(
                [self.command, "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                check=True
            )
            response = result.stdout.strip()
            logger.info(f"Ollama response: {response}")
            return response
        except subprocess.CalledProcessError as e:
            error_msg = f"Error running Ollama: {e.stderr.strip()}"
            logger.error(error_msg)
            return error_msg
        except FileNotFoundError:
            error_msg = "Ollama CLI not found. Ensure it is installed and in your PATH."
            logger.error(error_msg)
            return error_msg
