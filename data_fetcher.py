#!/usr/bin/env python3
# -------------------------------------------------------------------
# File Path: data_fetcher.py
# Description: Fetches stock data from various APIs, saves data into
#              a database, applies technical indicators, and handles
#              API rate limiting.
# -------------------------------------------------------------------

import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from aiohttp import ClientSession, ClientTimeout
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
import ta  # Technical Analysis library

# --------------------- Configuration and Setup ---------------------

# Load environment variables from .env file
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print("Environment variables loaded from .env")
else:
    print("Warning: .env file not found. Ensure environment variables are set.")

# Setup logging
def setup_logging(
    script_name: str,
    log_dir: Path,
    max_log_size: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 3,
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = RotatingFileHandler(
        log_dir / f"{script_name}.log",
        maxBytes=max_log_size,
        backupCount=backup_count
    )
    file_handler.setLevel(file_log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging(
    script_name="data_fetcher",
    log_dir=Path('./logs'),
    max_log_size=5 * 1024 * 1024,  # 5MB
    backup_count=3,
    console_log_level=logging.INFO,
    file_log_level=logging.DEBUG
)

# Database setup using SQLAlchemy
class DataStore:
    def __init__(self, database_url: str):
        try:
            self.engine = create_engine(database_url)
            logger.debug("Database engine created.")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            sys.exit(1)

    def save_data(self, df: pd.DataFrame, symbol: str):
        try:
            df.to_sql(symbol.lower(), con=self.engine, if_exists='append', index=True)
            logger.debug(f"Data saved to database table '{symbol.lower()}'.")
        except SQLAlchemyError as e:
            logger.error(f"Error saving data to database: {e}")

# Function to apply technical indicators
def apply_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if 'close' in df.columns:
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        logger.debug("Applied technical indicators: SMA_20, RSI.")
    else:
        logger.warning("Column 'close' not found in DataFrame. Technical indicators not applied.")
    return df

# Utility function to save DataFrame to CSV
def save_to_csv(df: pd.DataFrame, symbol: str, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{symbol}.csv"
    df.to_csv(file_path, index=True)
    logger.debug(f"Data saved to CSV at {file_path}.")

# --------------------- Unified Data Fetcher Class ---------------------

class UnifiedDataFetcher:
    POLYGON_BASE_URL = 'https://api.polygon.io/v2/aggs/ticker'
    ALPHA_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, interval: str):
        self.load_api_keys()
        self.interval = interval.lower()
        self.validate_interval()
        self.raw_csv_dir = Path('./data/real_time/raw')
        self.raw_csv_dir.mkdir(parents=True, exist_ok=True)

    def load_api_keys(self):
        self.polygon_api_key = os.getenv("POLYGONIO_API_KEY")
        self.alpha_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        missing_keys = []
        if not self.polygon_api_key:
            missing_keys.append("POLYGONIO_API_KEY")
        if not self.alpha_api_key:
            missing_keys.append("ALPHAVANTAGE_API_KEY")
        if not self.alpaca_api_key:
            missing_keys.append("ALPACA_API_KEY")
        if not self.alpaca_secret_key:
            missing_keys.append("ALPACA_SECRET_KEY")

        if missing_keys:
            logger.error(f"Missing API keys: {', '.join(missing_keys)}")
            sys.exit(1)

        logger.debug("All API keys loaded successfully.")

    def validate_interval(self):
        valid_intervals = ['day', 'minute', 'tick']
        if self.interval not in valid_intervals:
            logger.error(f"Unsupported interval '{self.interval}'. Supported intervals: {valid_intervals}")
            sys.exit(1)

    def construct_polygon_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        url = f"{self.POLYGON_BASE_URL}/{symbol}/range/1/{self.interval}/{start_date}/{end_date}?adjusted=true&apiKey={self.polygon_api_key}"
        logger.debug(f"Constructed Polygon API URL: {url}")
        return url

    async def fetch_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        async with ClientSession() as session:
            data, error = await self.fetch_from_polygon(symbol, start_date, end_date, session)
            if data is not None:
                return data

            data, error = await self.fetch_from_yfinance(symbol, start_date, end_date)
            if data is not None:
                return data

            logger.error(f"All attempts to fetch data for {symbol} failed.")
            return None

    async def fetch_from_polygon(self, symbol: str, start_date: str, end_date: str, session: ClientSession) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        url = self.construct_polygon_api_url(symbol, start_date, end_date)
        logger.info(f"Fetching data from Polygon for {symbol}")
        try:
            async with session.get(url) as response:
                content = await response.text()
                logger.debug(f"Polygon API Response Status: {response.status}, Content: {content}")

                if response.status == 200:
                    data = await response.json()
                    df = self.process_polygon_data(data, symbol)
                    return df, None
                elif response.status == 403:
                    logger.error(f"Unauthorized access for {symbol} from Polygon API.")
                    return None, 'Unauthorized'
                else:
                    logger.error(f"Polygon API request failed for {symbol} with status {response.status}.")
                    return None, f"Error {response.status}"
        except Exception as e:
            logger.error(f"Exception occurred while fetching from Polygon for {symbol}: {e}")
            return None, 'Exception'

    async def fetch_from_yfinance(self, symbol: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        logger.info(f"Fetching data from YFinance for {symbol}")
        try:
            interval = '1d' if self.interval == 'day' else '1m'
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.warning(f"No data returned from YFinance for {symbol}.")
                return None, 'NoData'

            df.reset_index(inplace=True)
            df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            df['symbol'] = symbol
            df.set_index('timestamp', inplace=True)
            return df, None
        except Exception as e:
            logger.error(f"Exception occurred while fetching from YFinance for {symbol}: {e}")
            return None, 'Exception'

    def process_polygon_data(self, data: dict, symbol: str) -> pd.DataFrame:
        results = data.get('results', [])
        if not results:
            logger.warning(f"No data found in Polygon response for {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df.set_index('timestamp', inplace=True)
        return df

# --------------------- Main Execution ---------------------

async def main():
    # Load configuration from environment variables
    interval = os.getenv("DATA_INTERVAL", 'day').lower()
    symbols = os.getenv("SYMBOLS", "AAPL,GOOGL,MSFT").split(",")
    start_date = os.getenv("START_DATE", "2023-01-01")
    end_date = os.getenv("END_DATE", "2023-12-31")

    # Log the configuration
    logger.info(f"Configuration - Interval: {interval}, Symbols: {symbols}, Start Date: {start_date}, End Date: {end_date}")

    # Initialize DataStore
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            logger.error("DATABASE_URL environment variable is missing.")
            sys.exit(1)
        store = DataStore(database_url)
    except Exception as e:
        logger.error(f"Failed to initialize DataStore: {e}")
        sys.exit(1)

    # Initialize DataFetcher
    fetcher = UnifiedDataFetcher(interval=interval)

    # Fetch data for each symbol
    for symbol in symbols:
        symbol = symbol.strip()
        logger.info(f"Fetching data for {symbol}")
        df = await fetcher.fetch_data(symbol, start_date, end_date)
        if df is not None and not df.empty:
            df = apply_technical_indicators(df)
            save_to_csv(df, symbol, fetcher.raw_csv_dir)
            store.save_data(df, symbol)
            logger.info(f"Data for {symbol} fetched and saved successfully.")
        else:
            logger.warning(f"No data fetched for {symbol}.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
