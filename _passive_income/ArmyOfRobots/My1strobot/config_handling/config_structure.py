# -------------------------------------------------------------------
# File Path: C:\TheTradingRobotPlug\Scripts\Utilities\config_handling\config_structure.py
# Description: Defines the configuration structure for the TradingRobotPlug project.
# This module outlines the default configuration values for various sections,
# including paths, API keys, logging settings, and database details, and includes 
# support for environment-based configuration and validation.
# -------------------------------------------------------------------

import os
from pathlib import Path

# -------------------------------------------------------------------
# Section 1: Environment-Specific Configurations
# -------------------------------------------------------------------

# Fetch environment type (development, production, etc.) from environment variables
ENV = os.getenv('TRADING_ROBOT_ENV', 'development')  # Default to 'development'

# Base directories (dynamic based on environment)
BASE_DIR = Path("C:/TheTradingRobotPlug")
if ENV == 'production':
    BASE_DIR = Path("/var/www/TradingRobotPlug")

# -------------------------------------------------------------------
# Section 2: Configuration Structure Definition
# -------------------------------------------------------------------

CONFIG_STRUCTURE = {
    "Paths": {
        "base_dir": str(BASE_DIR),
        "data_folder": str(BASE_DIR / "data"),
        "log_folder": str(BASE_DIR / "logs"),
        "db_path": str(BASE_DIR / "data" / "trading_data.db"),
        "csv_dir": str(BASE_DIR / "data" / "csv")
    },
    "API": {
        "alphavantage_api_key": os.getenv('ALPHAVANTAGE_API_KEY', "C6AG9NZX6QIPYTX4"),
        "alphavantage_base_url": "https://www.alphavantage.co/query",
        "polygonio_api_key": os.getenv('POLYGONIO_API_KEY', "your_polygon_api_key"),
        "nasdaq_api_key": os.getenv('NASDAQ_API_KEY', ""),
        "finnhub_api_key": os.getenv('FINNHUB_API_KEY', ""),
        "fred_api_key": os.getenv('FRED_API_KEY', ""),
        "timeout": os.getenv('API_TIMEOUT', "30"),
    },
    "Logging": {
        "log_file": str(BASE_DIR / "logs" / "trading_robot.log"),
        "real_time_log_file": str(BASE_DIR / "logs" / "real_time.log"),
        "alphavantage_log_file": str(BASE_DIR / "logs" / "alphavantage.log"),
        "polygon_log_file": str(BASE_DIR / "logs" / "polygon.log")
    },
    "Database": {
        "db_name": os.getenv('DB_NAME', "trading_data"),
        "db_user": os.getenv('DB_USER', "root"),
        "db_password": os.getenv('DB_PASSWORD', "password"),
    },
    "Data": {
        "real_time_raw_csv_dir": str(BASE_DIR / "data" / "real_time" / "raw"),
        "real_time_processed_csv_dir": str(BASE_DIR / "data" / "real_time" / "processed"),
        "alphavantage_raw_csv_dir": str(BASE_DIR / "data" / "alphavantage" / "raw"),
        "alphavantage_processed_csv_dir": str(BASE_DIR / "data" / "alphavantage" / "processed"),
        "polygon_raw_csv_dir": str(BASE_DIR / "data" / "polygon" / "raw"),
        "polygon_processed_csv_dir": str(BASE_DIR / "data" / "polygon" / "processed")
    }
}

# -------------------------------------------------------------------
# Section 3: Configuration Validation
# -------------------------------------------------------------------

def validate_config():
    """
    Validates the critical configuration values to ensure they are present and valid.
    Raises an error if any critical value is missing.
    """
    missing_values = []
    api_keys = ['alphavantage_api_key', 'polygonio_api_key']

    for key in api_keys:
        if not CONFIG_STRUCTURE['API'].get(key):
            missing_values.append(f"API.{key}")

    if missing_values:
        raise ValueError(f"Missing critical configuration values: {', '.join(missing_values)}")
    else:
        print("Configuration validation passed.")

# -------------------------------------------------------------------
# Section 4: Helper Functions to Access Config
# -------------------------------------------------------------------

def get_config_value(section, key, default=None):
    """
    Retrieves a configuration value from the CONFIG_STRUCTURE.

    Args:
        section (str): The section in the configuration structure (e.g., 'API', 'Paths').
        key (str): The key to retrieve the value for (e.g., 'alphavantage_api_key').
        default: The default value to return if the key does not exist.

    Returns:
        The configuration value or the default.
    """
    return CONFIG_STRUCTURE.get(section, {}).get(key, default)

def update_config_value(section, key, value):
    """
    Updates a configuration value in the CONFIG_STRUCTURE.

    Args:
        section (str): The section in the configuration structure (e.g., 'API', 'Paths').
        key (str): The key to update (e.g., 'alphavantage_api_key').
        value: The new value to set.
    """
    if section in CONFIG_STRUCTURE and key in CONFIG_STRUCTURE[section]:
        CONFIG_STRUCTURE[section][key] = value
        print(f"Updated {section}.{key} to {value}")
    else:
        print(f"Invalid section or key: {section}.{key}")

# -------------------------------------------------------------------
# Example Usage (Main Section)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Print out the configuration structure for debugging purposes
    import pprint
    pprint.pprint(CONFIG_STRUCTURE)

    # Example validation
    try:
        validate_config()
    except ValueError as e:
        print(f"Validation Error: {e}")

    # Example usage of helper functions
    api_key = get_config_value('API', 'alphavantage_api_key')
    print(f"Alphavantage API Key: {api_key}")

    # Update configuration dynamically
    update_config_value('API', 'alphavantage_api_key', 'NEW_API_KEY')
