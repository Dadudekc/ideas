# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/Scripts/Data_Fetchers/Stock_Data/data_fetcher.py
# Description: Fetches stock data from Polygon, Alpha Vantage, Alpaca, and YFinance APIs,
#              saves data into the database, applies technical indicators,
#              and handles API rate limiting.
# -------------------------------------------------------------------

import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from aiohttp import ClientSession, ClientTimeout
from dotenv import load_dotenv
import logging

# Dynamically set the project root based on the current file's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]

# Load environment variables dynamically from .env in the project root
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print("Environment variables loaded from .env")
else:
    print("Warning: .env file not found, ensure environment variables are set.")

# Define all relevant directories based on the new structure
directories = {
    'config': project_root / 'config',
    'data': project_root / 'data',
    'database': project_root / 'database',
    'deployment': project_root / 'deployment',
    'docs': project_root / 'docs',
    'reports': project_root / 'reports',
    'Scripts': project_root / 'Scripts',
    'model_training': project_root / 'Scripts' / 'model_training',
    'data_fetching': project_root / 'Scripts' / 'data_fetching',
    'data_processing': project_root / 'Scripts' / 'data_processing',
    'backtesting': project_root / 'Scripts' / 'backtesting',
    'trading_robot': project_root / 'Scripts' / 'trading_robot',
    'scheduler': project_root / 'Scripts' / 'scheduler',
    'utilities': project_root / 'Scripts' / 'utilities',
    'utilities_db': project_root / 'Scripts' / 'utilities' / 'db',
    'utilities_utils': project_root / 'Scripts' / 'utilities' / 'utils',
    'logs': project_root / 'logs',
    'results': project_root / 'results',
    'SavedModels': project_root / 'SavedModels',
}

# Add all directories to sys.path for module imports
for name, path in directories.items():
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    sys.path.append(str(path.resolve()))

# Import necessary utilities and classes
try:
    from config_handling.logging_setup import setup_logging
    from data.data_store import DataStore
    from data.data_fetch_utils import DataFetchUtils
    from main_indicators import apply_all_indicators
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Setup logging using logging_setup.py
logger = setup_logging(
    script_name="data_fetcher",
    log_dir=Path("C:/Projects/TradingRobotPlug/logs/DatabaseTests"),  # Specified log directory
    max_log_size=5 * 1024 * 1024,  # 5MB
    backup_count=3,
    console_log_level=logging.INFO,
    file_log_level=logging.DEBUG,
    feedback_loop_enabled=True
)

# Initialize DataStore with environment variables
try:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL is not set in environment variables.")
        raise ValueError("Missing DATABASE_URL.")
    store = DataStore(database_url=db_url, use_csv=False)
    logger.info("DataStore initialized successfully.")
except Exception as e:
    logger.error(f"Unexpected error initializing DataStore: {e}")

class UnifiedDataFetcher:
    """
    Fetches stock data from various APIs (Polygon, Alpha Vantage, Alpaca, YFinance),
    saves data into the database, applies technical indicators, and handles API rate limiting.
    """
    POLYGON_BASE_URL = 'https://api.polygon.io/v2/aggs/ticker'
    ALPHA_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, interval: str):
        """
        Initialize the UnifiedDataFetcher with the desired interval.

        :param interval: The data interval (e.g., 'day', 'minute', 'tick')
        """
        self.polygon_api_key = os.getenv("POLYGONIO_API_KEY")
        self.alpha_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self.alpaca_api_key = os.getenv("APCA_API_KEY_ID")
        self.alpaca_secret_key = os.getenv("APCA_API_SECRET_KEY")
        self.alpaca_base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

        if not self.polygon_api_key:
            logger.error("POLYGONIO_API_KEY is not set in environment variables.")
            raise ValueError("Missing POLYGONIO_API_KEY.")

        if not self.alpha_api_key:
            logger.error("ALPHAVANTAGE_API_KEY is not set in environment variables.")
            raise ValueError("Missing ALPHAVANTAGE_API_KEY.")

        if not self.alpaca_api_key or not self.alpaca_secret_key:
            logger.error("Alpaca API keys are not set in environment variables.")
            raise ValueError("Missing Alpaca API keys.")

        self.raw_csv_dir = Path(os.getenv("BASE_DIRECTORY", project_root / 'data')) / 'real_time' / 'raw'
        self.raw_csv_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval.lower()
        self.valid_intervals = ['day', 'minute', 'tick']

        if self.interval not in self.valid_intervals:
            logger.error(f"Unsupported interval '{self.interval}'. Supported intervals: {self.valid_intervals}")
            raise ValueError(f"Unsupported interval '{self.interval}'.")

    def construct_polygon_api_url(self, symbol: str, start_date: str, end_date: str) -> str:
        if self.interval == 'tick':
            return f"https://api.polygon.io/v2/ticks/stocks/trades/{symbol}/{start_date}/{end_date}?apiKey={self.polygon_api_key}"
        else:
            return f"{self.POLYGON_BASE_URL}/{symbol}/range/1/{self.interval}/{start_date}/{end_date}?adjusted=true&apiKey={self.polygon_api_key}"

    def construct_alpha_api_url(self, symbol: str) -> str:
        return f"{self.ALPHA_BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.alpha_api_key}"

    def construct_alpaca_client(self):
        from alpaca_trade_api.rest import REST
        return REST(self.alpaca_api_key, self.alpaca_secret_key, self.alpaca_base_url)

    async def fetch_data_for_symbol(self, symbol: str, start_date: str, end_date: str, session: ClientSession) -> Optional[pd.DataFrame]:
        """
        Fetches data for a symbol from multiple APIs, attempting in order:
        Polygon, Alpha Vantage, Alpaca, and YFinance.
        """
        # Attempt to fetch from Polygon
        data = await self.fetch_from_polygon(symbol, start_date, end_date, session)
        if data is not None and not data.empty:
            return data

        # Attempt to fetch from Alpha Vantage
        data = await self.fetch_from_alpha_vantage(symbol, session)
        if data is not None and not data.empty:
            return data

        # Attempt to fetch from Alpaca
        data = await self.fetch_from_alpaca(symbol, start_date, end_date)
        if data is not None and not data.empty:
            return data

        # Attempt to fetch from YFinance
        data = await self.fetch_from_yfinance(symbol, start_date, end_date)
        if data is not None and not data.empty:
            return data

        logger.error(f"All attempts to fetch data for {symbol} failed.")
        return None

    async def fetch_from_polygon(self, symbol, start_date, end_date, session):
        try:
            url = self.construct_polygon_api_url(symbol, start_date, end_date)
            logger.debug(f"Fetching data from Polygon URL: {url}")
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Polygon API response status {response.status} for symbol {symbol}.")
                    raise RuntimeError(f"Polygon API response status {response.status}.")
                data = await response.json()

            if 'results' not in data:
                logger.warning(f"No 'results' in Polygon API response for {symbol}.")
                raise ValueError("Invalid Polygon API response structure.")

            df = self.process_polygon_data(data, symbol)
            DataFetchUtils.save_to_csv(df, symbol, self.raw_csv_dir, logger)
            return df

        except Exception as e:
            logger.error(f"Polygon API fetch failed for {symbol}: {e}")
            return None

    async def fetch_from_alpha_vantage(self, symbol, session):
        try:
            url = self.construct_alpha_api_url(symbol)
            logger.debug(f"Fetching data from Alpha Vantage URL: {url}")
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Alpha Vantage API response status {response.status} for symbol {symbol}.")
                    raise RuntimeError(f"Alpha Vantage API response status {response.status}.")
                data = await response.json()

            if "Time Series (1min)" not in data:
                logger.error(f"Unexpected Alpha Vantage API response for {symbol}.")
                raise ValueError("Invalid Alpha Vantage API response structure.")

            df = self.process_alpha_data(data, symbol)
            DataFetchUtils.save_to_csv(df, symbol, self.raw_csv_dir, logger)
            return df

        except Exception as e:
            logger.error(f"Alpha Vantage API fetch failed for {symbol}: {e}")
            return None

    async def fetch_from_alpaca(self, symbol, start_date, end_date):
        try:
            logger.debug(f"Fetching data from Alpaca API for {symbol}")
            api = self.construct_alpaca_client()
            from alpaca_trade_api.rest import TimeFrame

            timeframe = TimeFrame.Day if self.interval == 'day' else TimeFrame.Minute

            bars = api.get_bars(
                symbol,
                timeframe,
                start=pd.to_datetime(start_date).isoformat(),
                end=pd.to_datetime(end_date).isoformat(),
            ).df

            if bars.empty:
                logger.warning(f"No data returned from Alpaca for {symbol}.")
                return None

            bars.reset_index(inplace=True)
            bars.rename(columns={'timestamp': 'date'}, inplace=True)
            bars['symbol'] = symbol
            bars.set_index('date', inplace=True)
            DataFetchUtils.save_to_csv(bars, symbol, self.raw_csv_dir, logger)
            return bars

        except Exception as e:
            logger.error(f"Alpaca API fetch failed for {symbol}: {e}")
            return None

    async def fetch_from_yfinance(self, symbol, start_date, end_date):
        try:
            logger.debug(f"Fetching data from YFinance for {symbol}")
            import yfinance as yf
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval='1d' if self.interval == 'day' else '1m')

            if df.empty:
                logger.warning(f"No data returned from YFinance for {symbol}.")
                return None

            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            df['symbol'] = symbol
            df.set_index('date', inplace=True)
            DataFetchUtils.save_to_csv(df, symbol, self.raw_csv_dir, logger)
            return df

        except Exception as e:
            logger.error(f"YFinance fetch failed for {symbol}: {e}")
            return None

    def process_polygon_data(self, data: dict, symbol: str) -> pd.DataFrame:
        results = data.get('results', [])
        df = pd.DataFrame(results)
        if 't' in df.columns:
            df.rename(columns={'t': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        else:
            logger.error(f"Polygon data for {symbol} is missing 't' (timestamp) column.")
            return pd.DataFrame()

        df['symbol'] = symbol
        required_columns = ['h', 'l', 'c', 'v', 'o']
        if all(col in df.columns for col in required_columns):
            df.rename(columns={
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'o': 'open'
            }, inplace=True)
        else:
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Polygon data for {symbol} is missing columns: {', '.join(missing)}.")
            return pd.DataFrame()

        return df

    def process_alpha_data(self, data: dict, symbol: str) -> pd.DataFrame:
        time_series = data.get('Time Series (1min)', {})
        results = [
            {
                'timestamp': timestamp,
                'open': float(values["1. open"]),
                'high': float(values["2. high"]),
                'low': float(values["3. low"]),
                'close': float(values["4. close"]),
                'volume': int(values["5. volume"])
            }
            for timestamp, values in time_series.items()
        ]
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol
        return df

    async def fetch_data_for_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Optional[pd.DataFrame]]:
        timeout = ClientTimeout(total=300)  # Increased timeout for multiple requests
        async with ClientSession(timeout=timeout) as session:
            tasks = [
                self.fetch_data_for_symbol(symbol.strip(), start_date, end_date, session)
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            data_dict = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"Exception occurred while fetching data for {symbol}: {result}")
                elif result is not None and not result.empty:
                    df_with_indicators = apply_all_indicators(result)
                    data_dict[symbol] = df_with_indicators
                    try:
                        store.save_data(df_with_indicators, symbol=symbol, apply_indicators=False)
                        logger.info(f"Data with indicators saved to the database for {symbol}.")
                    except Exception as e:
                        logger.error(f"Error saving data for {symbol} to SQL database: {e}")
                else:
                    logger.warning(f"No data returned or empty DataFrame for {symbol}.")
            return data_dict

async def main():
    try:
        interval = os.getenv("DATA_INTERVAL", 'day').lower()
        symbols = os.getenv("SYMBOLS", "AAPL,GOOGL,MSFT").split(",")
        start_date = os.getenv("START_DATE", "2023-01-01")
        end_date = os.getenv("END_DATE", "2023-12-31")
    except Exception as e:
        logger.error(f"Error reading environment variables: {e}")
        return

    try:
        fetcher = UnifiedDataFetcher(interval=interval)
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        return

    logger.info(f"Starting data fetch for symbols: {symbols} with interval: {interval}")

    data_dict = await fetcher.fetch_data_for_multiple_symbols(symbols, start_date, end_date)

    if data_dict:
        logger.info("Data fetching and processing completed successfully.")
    else:
        logger.warning("No data was fetched or processed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
