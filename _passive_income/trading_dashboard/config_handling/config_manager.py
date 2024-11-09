# Filename: config_manager.py
# Description: Manages configuration and environment variables for flexible projects.
#              Supports loading from environment variables, .env files, YAML, JSON, and TOML files.
#              Provides type casting, validation, and dynamic reloading capabilities.

import os
import yaml
import json
import toml
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
import logging
from typing import Any, Optional, List, Dict, Union, Type
import threading

class ConfigManager:
    def __init__(self, 
                 config_files: Optional[List[Path]] = None, 
                 env_file: Optional[Path] = None,
                 required_keys: Optional[List[str]] = None, 
                 logger: Optional[logging.Logger] = None,
                 custom_root: Optional[Path] = None):
        """
        Initializes the ConfigManager and loads configurations from various sources.

        Args:
            config_files (list): List of configuration files to load (YAML, JSON, TOML).
            env_file (Path): Path to the .env file. If not provided, attempts to load from project root or custom root.
            required_keys (list): List of required configuration keys to check.
            logger (Logger): Logger object to log messages (optional).
            custom_root (Path): Allows setting a custom root path for different project structures.
        """
        self.config = defaultdict(dict)
        self.logger = logger or self.setup_logger("ConfigManager")
        self.cache = {}
        self.lock = threading.Lock()
        self.required_keys = required_keys or []

        # Dynamically set project root based on custom path or default to this file's location
        script_dir = Path(__file__).resolve().parent
        self.project_root = custom_root or script_dir.parents[2]

        # Determine the .env file path with fallback options
        env_path = env_file or (self.project_root / '.env')
        if env_path.exists():
            self._load_env(env_path)
        else:
            self.logger.warning(f"No .env file found at {env_path}. Environment variables will be used as-is.")

        # Load configurations from specified files
        if config_files:
            self.load_configurations(config_files)

        # Validate required keys
        self.check_missing_keys()

    def load_configurations(self, config_files: List[Path]):
        """Loads configuration data from various file formats."""
        for config_file in config_files:
            file_ext = config_file.suffix.lower()
            if not config_file.exists():
                self.logger.warning(f"Config file does not exist: {config_file}")
                continue
            loader = {
                '.yaml': self._load_yaml,
                '.yml': self._load_yaml,
                '.json': self._load_json,
                '.toml': self._load_toml
            }.get(file_ext)
            if loader:
                loader(config_file)
            else:
                self.logger.warning(f"Unsupported config file format: {config_file}")

    def get(self, key: str, default: Any = None, required: bool = False, value_type: Optional[Type] = None) -> Any:
        """
        Retrieves a configuration value with optional type casting.

        Args:
            key (str): The key to retrieve.
            default (Any): Default value if the key is not found.
            required (bool): Raises an error if True and the key is missing.
            value_type (Type): The expected type of the configuration value.

        Returns:
            Any: The retrieved configuration value, optionally cast to the specified type.
        """
        full_key = key.lower()
        with self.lock:
            value = self.cache.get(full_key) or os.getenv(full_key.upper()) or self.config.get(full_key) or default
            if value is None and required:
                raise ValueError(f"Required configuration '{key}' is missing.")
            return self._cast_type(value, value_type)

    def _cast_type(self, value: Any, value_type: Optional[Type]) -> Any:
        """Cast value to specified type, with special handling for bool, list, and dict types."""
        if value is None or value_type is None:
            return value
        try:
            if value_type == bool:
                return self._str_to_bool(value)
            elif value_type == list and isinstance(value, str):
                return [item.strip() for item in value.split(',')]
            elif value_type == dict and isinstance(value, str):
                return json.loads(value)
            return value_type(value)
        except Exception as e:
            self.logger.error(f"Failed to cast '{value}' to {value_type}: {e}")
            raise TypeError(f"Failed to cast '{value}' to {value_type}: {e}")
        
    def _load_env(self, env_path: Path):
        """Loads environment variables from a .env file."""
        print(f"Attempting to load environment variables from: {env_path}")  # Debug statement
        load_dotenv(dotenv_path=env_path, override=True)  # Set override=True
        self.logger.info(f"Loaded environment variables from {env_path}")

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> None:
        """Flattens nested dictionaries for easy access."""
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                self._flatten_dict(v, new_key, sep=sep)
            else:
                self.config[new_key.lower()] = v

    def check_missing_keys(self):
        """Ensures all required configuration keys are present."""
        missing_keys = [key.upper() for key in self.required_keys if self.get(key) is None]
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    def get_all(self) -> Dict[str, Any]:
        """
        Returns all configurations as a dictionary, combining data from config files and environment variables.

        Returns:
            Dict[str, Any]: Combined configuration data.
        """
        all_configs = dict(self.config)
        for key, value in os.environ.items():
            all_configs[key.lower()] = value
        return all_configs

    def get_db_url(self) -> str:
        """
        Constructs and returns the database URL based on the configuration values.

        Returns:
            str: The constructed database URL.
        """
        db_user = self.get('POSTGRES_USER', required=True)
        db_password = self.get('POSTGRES_PASSWORD', required=True)
        db_host = self.get('POSTGRES_HOST', required=True)
        db_port = self.get('POSTGRES_PORT', required=True)
        db_name = self.get('POSTGRES_DBNAME', required=True)
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def get_async_db_url(self) -> str:
        """
        Constructs and returns the asynchronous database URL.

        Returns:
            str: The constructed asynchronous database URL.
        """
        db_user = self.get('POSTGRES_USER', required=True)
        db_password = self.get('POSTGRES_PASSWORD', required=True)
        db_host = self.get('POSTGRES_HOST', required=True)
        db_port = self.get('POSTGRES_PORT', required=True)
        db_name = self.get('POSTGRES_DBNAME', required=True)
        return f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def reload_configurations(self, config_files: Optional[List[Path]] = None, env_file: Optional[Path] = None):
        """
        Reloads configurations from specified files and environment variables.

        Args:
            config_files (Optional[List[Path]]): List of configuration file paths to reload.
            env_file (Optional[Path]): Path to the .env file to reload.
        """
        with self.lock:
            if config_files:
                self.load_configurations(config_files)
            if env_file:
                self._load_env(env_file)
            self.cache.clear()
            self.check_missing_keys()
            self.logger.info("Configurations reloaded successfully.")

    def list_configurations(self, mask_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Lists all loaded configurations, optionally masking sensitive keys.

        Args:
            mask_keys (Optional[List[str]]): List of keys to mask in the output.

        Returns:
            Dict[str, Any]: Dictionary of all configurations with masked values where applicable.
        """
        all_configs = self.get_all()
        if mask_keys:
            for key in mask_keys:
                key_lower = key.lower()
                if key_lower in all_configs:
                    all_configs[key_lower] = "*****"  # Mask sensitive information
        return all_configs

    @staticmethod
    def setup_logger(log_name: str) -> logging.Logger:
        """
        Sets up a logger that writes to both the console and a file.

        Args:
            log_name (str): Name of the logger.

        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create file handler
            script_dir = Path(__file__).resolve().parent
            log_dir = script_dir.parents[2] / 'logs' / log_name
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{log_name}.log")
            file_handler.setLevel(logging.DEBUG)

            # Create formatters and add them to the handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers to the logger
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger
from pathlib import Path
import logging

# Step 1: Initialize ConfigManager
# Specify paths to configuration files if available
config_files = [Path('config.yaml'), Path('config.json')]
env_file_path = Path('C:/Projects/#TODO/ideas/.env')

# Optional: Set up a custom logger
logger = logging.getLogger("MyAppLogger")
logger.setLevel(logging.INFO)

# Initialize the ConfigManager
config_manager = ConfigManager(
    config_files=config_files,
    env_file=env_file_path,
    required_keys=['POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DBNAME'],
    logger=logger
)

# Step 2: Retrieve a Configuration Value
db_user = config_manager.get('POSTGRES_USER')
db_password = config_manager.get('POSTGRES_PASSWORD')
db_host = config_manager.get('POSTGRES_HOST')
db_port = config_manager.get('POSTGRES_PORT', default='5432')  # Default to 5432 if not set
db_name = config_manager.get('POSTGRES_DBNAME')

# Step 3: Construct Database URL
db_url = config_manager.get_db_url()
print(f"Database URL: {db_url}")

# Example Output of Configuration Listing
all_configs = config_manager.list_configurations(mask_keys=['POSTGRES_PASSWORD'])
print("All Configurations (masked):", all_configs)
