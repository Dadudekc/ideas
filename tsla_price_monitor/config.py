# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    # Database configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///tsla_monitor.db")

    # Robinhood credentials
    ROBINHOOD_USERNAME: str = os.getenv("ROBINHOOD_USERNAME")
    ROBINHOOD_PASSWORD: str = os.getenv("ROBINHOOD_PASSWORD")

    # Alpaca API credentials
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Ollama configuration
    OLLAMA_COMMAND: str = os.getenv("OLLAMA_COMMAND", "ollama")

    # Logging configuration
    LOG_FILE: str = os.getenv("LOG_FILE", "tsla_price_monitor.log")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
