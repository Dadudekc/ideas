# utils.py

import os
import yaml
import logging

def load_config(config_file):
    """
    Load configuration from a YAML file. If the file does not exist, a default configuration
    is created, saved, and returned.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data loaded from the file.
    """
    # Default configuration
    default_config = {
        "mode": "backtest",
        "symbol": "TSLA",
        "timeframe": "1D",
        "limit": 1000,
        "api_key": "",
        "api_secret": "",
        "base_url": "https://paper-api.alpaca.markets",
        "strategy": {
            "vwap_session": "RTH",  # String parameter
            "ema_length": 8,
            "rsi_length": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_length": 14,
            "volume_threshold_length": 20,
            "atr_length": 14,
            "risk_percent": 0.5,
            "profit_target_percent": 15.0,
            "stop_multiplier": 2.0,
            "trail_multiplier": 1.5
        }
    }

    # Check if configuration file exists
    if not os.path.exists(config_file):
        try:
            # Create and save the default configuration
            with open(config_file, "w") as file:
                yaml.dump(default_config, file, default_flow_style=False)
            logging.info(f"Default configuration created at {config_file}.")
        except Exception as e:
            logging.error(f"Error creating default configuration: {e}")
            raise

    # Load the configuration from the file
    try:
        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)
            # Validate configuration
            if not isinstance(config_data, dict):
                raise ValueError("Configuration file is malformed or empty.")
            return config_data
    except Exception as e:
        logging.error(f"Error loading configuration file {config_file}: {e}")
        raise

def save_config(config_file, config_data):
    """
    Save configuration data to a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.
        config_data (dict): Configuration data to be saved.

    Returns:
        None
    """
    try:
        with open(config_file, "w") as file:
            yaml.dump(config_data, file, default_flow_style=False)
        logging.info(f"Configuration saved successfully to {config_file}.")
    except Exception as e:
        logging.error(f"Error saving configuration to {config_file}: {e}")
        raise

def setup_logger():
    """
    Set up a logger that logs to both console and GUI.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("TSLA_Trading_Bot")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
    logger.handlers = []  # Reset handlers

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Capture all levels
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
