# standalone_trading_robot.py

import os
import sys
import yaml
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import aiohttp
import pandas as pd
import ta
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf
from logging.handlers import RotatingFileHandler

# -------------------------------------------------------------------
# Section 1: Configuration Handling
# -------------------------------------------------------------------

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file. If the file does not exist, create a default configuration,
    save it to the file, and return it.
    """
    # Default configuration
    default_config = {
        "symbol": "TSLA",
        "timeframe": "1d",
        "limit": 1000,
        "strategy": {
            "ema_length": 8,
            "rsi_length": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
        },
        "log_dir": "./logs",
        "logging": {
            "log_level": "INFO",
            "log_file": "robot.log",
            "max_log_size": 5242880,  # 5MB
            "backup_count": 2
        },
        "data_fetching": {
            "fetch_retries": 3,
            "backoff_factor": 2
        },
    }

    # Ensure the config directory exists
    config_path = Path(config_file)
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create default config file if it doesn't exist
    if not config_path.exists():
        with open(config_path, "w") as file:
            yaml.dump(default_config, file, default_flow_style=False)
        print(f"Default configuration created at {config_file}. Please review and update it as necessary.")

    # Load the configuration from the file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Validate that the configuration is a dictionary
    if not isinstance(config, dict):
        raise ValueError("Configuration file is malformed or empty.")

    # Load environment variables and override defaults
    load_dotenv()
    env_config = {key: os.getenv(key) for key in os.environ if key in config}
    config.update(env_config)

    return config

# -------------------------------------------------------------------
# Section 2: Logging Setup
# -------------------------------------------------------------------

def setup_logger(
    name: str,
    log_dir: str,
    log_file: str,
    max_log_size: int,
    backup_count: int,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Set up a logger that logs to both console and file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))  # Set logging level

    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        Path(log_dir) / log_file,
        maxBytes=max_log_size,
        backupCount=backup_count
    )
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))  # Match log level
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# -------------------------------------------------------------------
# Section 3: Data Fetching Utilities
# -------------------------------------------------------------------

class DataFetchUtils:
    def __init__(self, logger: logging.Logger):
        """
        Initialize the DataFetchUtils class.
        """
        self.logger = logger

    async def fetch_stock_data_async(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Asynchronously fetches stock data using yfinance.
        """
        self.logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date} at {interval} intervals.")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty:
            self.logger.error(f"No data fetched for {ticker}.")
            return pd.DataFrame()

        data.reset_index(inplace=True)
        # Convert all column names to lowercase
        data.columns = [col.lower() for col in data.columns]
        # Rename 'adj close' to 'adj_close' for consistency
        data.rename(columns={'adj close': 'adj_close'}, inplace=True)
        data['symbol'] = ticker

        return data


# -------------------------------------------------------------------
# Section 4: Strategy Class
# -------------------------------------------------------------------

class Strategy:
    """
    Strategy class encapsulates the parameters and logic required for trading strategies.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Strategy class with provided parameters.
        """
        strategy_params = config.get("strategy", {})
        self.ema_length = strategy_params.get("ema_length", 8)
        self.rsi_length = strategy_params.get("rsi_length", 14)
        self.rsi_overbought = strategy_params.get("rsi_overbought", 70)
        self.rsi_oversold = strategy_params.get("rsi_oversold", 30)
        self.logger = logger

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading indicators on the provided DataFrame.
        """
        self.logger.info("Calculating indicators.")
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=self.ema_length).ema_indicator()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_length).rsi()
        return df


    def evaluate(self, df: pd.DataFrame) -> List[str]:
        """
        Evaluate the latest indicators and generate trading signals.
        """
        self.logger.info("Evaluating strategy based on indicators.")
        signals = []
        if df.empty:
            self.logger.warning("DataFrame is empty. No signals generated.")
            return signals

        latest = df.iloc[-1]
        rsi_value = latest['rsi']
        self.logger.info(f"Latest RSI value: {rsi_value}")

        # Example enhanced evaluation logic
        if rsi_value < self.rsi_oversold:
            signals.append('BUY')
            self.logger.info("BUY signal generated: RSI indicates oversold conditions.")
        elif rsi_value > self.rsi_overbought:
            signals.append('SELL')
            self.logger.info("SELL signal generated: RSI indicates overbought conditions.")
        else:
            signals.append('HOLD')
            self.logger.info(f"HOLD signal generated: RSI is neutral ({rsi_value}).")


        return signals


# -------------------------------------------------------------------
# Section 5: Main Execution Flow
# -------------------------------------------------------------------

def main():
    # Setup and Initialization
    try:
        config = load_config("1config.yaml")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)

    logger = setup_logger("TradingRobot", config['log_dir'], **config['logging'])

    # Initialize utilities
    data_fetcher = DataFetchUtils(logger)

    # Configuration settings
    symbol = config.get("symbol", "TSLA")
    timeframe = config.get("timeframe", "1d")
    limit = config.get("limit", 30)

    # Validate timeframe
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    if timeframe not in valid_intervals:
        logger.error(f"Invalid interval '{timeframe}'. Must be one of {valid_intervals}.")
        sys.exit(1)

    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=limit)).strftime('%Y-%m-%d')

    # Fetch data
    df = asyncio.run(
        data_fetcher.fetch_stock_data_async(
            ticker=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=timeframe
        )
    )

    if df.empty:
        logger.error(f"No data available for {symbol}. Exiting.")
        sys.exit(1)

    # Initialize and run strategy
    strategy = Strategy(config, logger)
    df_with_indicators = strategy.calculate_indicators(df)

    # Evaluate strategy
    signals = strategy.evaluate(df_with_indicators)
    if signals:
        logger.info(f"Generated signals: {signals}")
        print(f"Generated signals: {signals}")
    else:
        logger.info("No trading signals generated.")
        print("No trading signals generated.")

if __name__ == "__main__":
    main()
