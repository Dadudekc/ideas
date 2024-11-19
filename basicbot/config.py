from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict
import yaml
from pathlib import Path
from dotenv import load_dotenv
import os


class LoggingConfig(BaseModel):
    log_dir: str = Field(default="./logs", description="Directory to store log files.")
    log_level: str = Field(default="INFO", description="Logging level.")
    log_file: str = Field(default="robot.log", description="Log file name.")
    max_log_size: int = Field(default=5242880, description="Maximum log file size (in bytes).")  # 5MB
    backup_count: int = Field(default=2, description="Number of backup log files to retain.")


class StrategyConfig(BaseModel):
    ema_length: int = Field(default=8, ge=1, description="Length of EMA calculation.")
    rsi_length: int = Field(default=14, ge=1, description="Length of RSI calculation.")
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0, description="RSI overbought threshold.")
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0, description="RSI oversold threshold.")


class DataFetchingConfig(BaseModel):
    api_key: str = Field(default="your_default_api_key", description="API key for data fetching.")
    fetch_retries: int = Field(default=3, ge=0, description="Number of retry attempts for data fetching.")
    backoff_factor: float = Field(default=2.0, ge=0.0, description="Backoff factor for retry delays.")
    start_date: str = Field(default="2023-01-01", description="Default start date for data fetching.")
    end_date: str = Field(default="2023-12-31", description="Default end date for data fetching.")
    interval: str = Field(default="1d", description="Default interval for data fetching.")


class ConfigModel(BaseModel):
    symbol: str = Field(default="TSLA", description="Default stock symbol to trade.")
    limit: int = Field(default=1000, ge=1, description="Maximum number of data points to fetch.")
    strategy: StrategyConfig = StrategyConfig()
    logging: LoggingConfig = LoggingConfig()
    data_fetching: DataFetchingConfig = DataFetchingConfig()
    notification: Dict[str, Any] = Field(default_factory=dict, description="Notification configuration.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        """
        return self.model_dump()


def recursive_env_override(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Recursively override configuration with environment variables.
    """
    for key, value in config.items():
        env_key = f"{prefix}{key}".upper()
        if isinstance(value, dict):
            config[key] = recursive_env_override(value, prefix=f"{env_key}_")
        else:
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    config[key] = type(value)(env_value)  # Cast to original type
                except ValueError:
                    print(f"Environment variable {env_key} could not be cast to {type(value)}. Skipping.")
    return config


def load_config(config_file: str) -> ConfigModel:
    """
    Load and validate configuration from a YAML file.
    """
    config_path = Path(config_file)

    # Ensure the config directory exists
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load environment variables
    load_dotenv()

    # Create default config if not exists
    if not config_path.exists():
        default_config = ConfigModel().model_dump()
        with open(config_file, "w") as file:
            yaml.dump(default_config, file, default_flow_style=False)
        print(f"Default configuration created at {config_file}. Please review and update it as necessary.")

    # Load config from file
    try:
        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error loading configuration file: {e}")

    # Override with environment variables
    config_data = recursive_env_override(config_data)

    # Validate and return config
    try:
        return ConfigModel(**config_data)
    except ValidationError as e:
        raise RuntimeError(f"Configuration validation failed: {e}")
