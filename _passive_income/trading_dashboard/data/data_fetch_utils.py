# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/Scripts/Utilities/data_fetch_utils.py
# Description: A utility class for fetching and processing data from
#              multiple financial APIs (Alpha Vantage, Polygon, Yahoo Finance,
#              Finnhub, and Alpaca), with integrated logging and retry logic.
#              API keys and sensitive information are loaded from a .env file.
#              Paths are dynamically determined for portability.
# -------------------------------------------------------------------

import os
import pandas as pd
import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any
from aiohttp import ClientSession, ClientConnectionError, ContentTypeError
from datetime import datetime
from pathlib import Path
import sys
from dotenv import load_dotenv  # For loading environment variables
import alpaca_trade_api as tradeapi  # For Alpaca API

# -------------------------------------------------------------------
# Section 0: Setup and Configuration
# -------------------------------------------------------------------

# Determine the project root dynamically based on the script's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent  # Adjust based on your project structure

# Define the path to the .env file (assumed to be at project root)
env_path = project_root / '.env'
if not env_path.exists():
    raise FileNotFoundError(f".env file not found at {env_path}")

# Load environment variables from the .env file
load_dotenv(dotenv_path=env_path)

# Import the centralized logging setup
# Adjust the import path based on your project structure
try:
    from config_handling.logging_setup import setup_logging
except ModuleNotFoundError:
    # Handle the case where the script is run from a different location
    sys.path.append(str(project_root / 'Scripts' / 'Utilities'))
    from config_handling.logging_setup import setup_logging

# Define base directories for logs and data using environment variables or default to project directories
base_log_dir = Path(os.getenv('LOG_DIR', project_root / 'logs'))
base_data_dir = Path(os.getenv('DATA_DIR', project_root / 'data'))

# Ensure the log directory exists
log_dir = base_log_dir / 'Utilities'
log_dir.mkdir(parents=True, exist_ok=True)

# Initialize Logging using setup_logging.py
logger = setup_logging(
    script_name="data_fetch_utils",
    log_dir=log_dir,
    max_log_size=5 * 1024 * 1024,  # 5 MB
    backup_count=3,
    console_log_level=logging.INFO,
    file_log_level=logging.DEBUG,
    feedback_loop_enabled=True
)

# -------------------------------------------------------------------
# Section 1: Alpaca API Initialization
# -------------------------------------------------------------------

def initialize_alpaca():
    """
    Initializes Alpaca API client using credentials from environment variables.

    :return:
        alpaca_api (tradeapi.REST): Initialized Alpaca API client
    """
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://data.alpaca.markets')  # Default to data URL

    if not all([api_key, secret_key]):
        logger.error("Alpaca API credentials are not fully set in the environment variables.")
        raise EnvironmentError("Missing Alpaca API credentials.")

    logger.info("Initializing Alpaca API with provided credentials.")
    alpaca_api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
    return alpaca_api

# -------------------------------------------------------------------
# Section 2: DataFetchUtils Class Definition
# -------------------------------------------------------------------
class DataFetchUtils:
    def __init__(self):
        """
        Initialize the DataFetchUtils class.
        All configurations are loaded via environment variables.
        """
        self.logger = logging.getLogger('DataFetchUtils')

    # -------------------------------------------------------------------
    # Section 3: Fetching Data with Retry Logic
    # -------------------------------------------------------------------

    async def fetch_with_retries(self, url: str, headers: Dict[str, str], session: ClientSession, retries: int = 3) -> Optional[Any]:
        """
        Fetches data from the provided URL with retries on failure.

        :param url: The API URL to fetch.
        :param headers: Headers to include in the request (e.g., API keys).
        :param session: An active ClientSession.
        :param retries: Number of retry attempts in case of failure.
        :return: Parsed JSON data as a dictionary or list, or None if failed.
        """
        for attempt in range(retries):
            try:
                self.logger.debug(f"Fetching data from URL: {url}, Attempt: {attempt + 1}")
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"Fetched data: {data}")
                        return data
                    elif response.status == 429:
                        self.logger.warning(f"Rate limit reached. Waiting before retry. Attempt {attempt + 1}/{retries}")
                        await asyncio.sleep(60)  # Wait for 60 seconds before retrying
                    else:
                        error_message = await response.text()
                        self.logger.error(f"Failed to fetch data from {url}. Status: {response.status}, Message: {error_message}")
            except ClientConnectionError as e:
                self.logger.error(f"ClientConnectionError: {e}")
            except ContentTypeError as e:
                self.logger.error(f"ContentTypeError: {e}")
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout error fetching data from {url}. Retrying ({attempt + 1}/{retries})...")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
            
            await asyncio.sleep(2 ** attempt)  # Exponential backoff for retries

        self.logger.error(f"Failed to fetch data after {retries} attempts: {url}")
        return None

    # -------------------------------------------------------------------
    # Section 4: Fetching Real-Time Stock Quotes
    # -------------------------------------------------------------------
    
    async def fetch_finnhub_quote(self, symbol: str, session: ClientSession) -> Optional[Dict[str, Any]]:
        """
        Fetches the latest stock quote data for a given symbol from Finnhub.

        :param symbol: Stock symbol (e.g., 'AAPL').
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a dictionary or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables.")
            raise EnvironmentError("Missing Finnhub API key.")

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub quote for symbol: {symbol}")

        return await self.fetch_with_retries(url, headers={}, session=session)

    # -------------------------------------------------------------------
    # Section 5: Fetching Basic Financial Metrics
    # -------------------------------------------------------------------
    
    async def fetch_finnhub_metrics(self, symbol: str, session: ClientSession) -> Optional[Dict[str, Any]]:
        """
        Fetches basic financial metrics for a given symbol from Finnhub.

        :param symbol: Stock symbol (e.g., 'AAPL').
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a dictionary or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables.")
            raise EnvironmentError("Missing Finnhub API key.")

        url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub financial metrics for symbol: {symbol}")

        return await self.fetch_with_retries(url, headers={}, session=session)

    # -------------------------------------------------------------------
    # Section 6: Fetching Stock Symbols for an Exchange
    # -------------------------------------------------------------------
    
    async def fetch_finnhub_symbols(self, exchange: str, session: ClientSession) -> Optional[list]:
        """
        Fetches available stock symbols for a given exchange from Finnhub.

        :param exchange: The stock exchange (e.g., 'US' for United States).
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a list of dictionaries or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables.")
            raise EnvironmentError("Missing Finnhub API key.")

        url = f"https://finnhub.io/api/v1/stock/symbol?exchange={exchange}&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub stock symbols for exchange: {exchange}")

        return await self.fetch_with_retries(url, headers={}, session=session)

    # -------------------------------------------------------------------
    # Section 7: Fetching Company News
    # -------------------------------------------------------------------
    
    async def fetch_company_news(self, symbol: str, from_date: str, to_date: str, session: ClientSession) -> Optional[list]:
        """
        Fetches recent news articles for a given symbol from Finnhub.

        :param symbol: Stock symbol (e.g., 'AAPL').
        :param from_date: Start date in YYYY-MM-DD format.
        :param to_date: End date in YYYY-MM-DD format.
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a list of dictionaries or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables.")
            raise EnvironmentError("Missing Finnhub API key.")

        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub company news for symbol: {symbol} from {from_date} to {to_date}")

        return await self.fetch_with_retries(url, headers={}, session=session)

    # -------------------------------------------------------------------
    # Section 8: Converting JSON to DataFrame
    # -------------------------------------------------------------------
    
    def convert_json_to_dataframe(self, data: Any, api_source: str) -> pd.DataFrame:
        """
        Converts JSON data to a pandas DataFrame for multiple APIs.

        :param data: Fetched JSON data from an API.
        :param api_source: Source API to determine how to parse the data.
        :return: A pandas DataFrame.
        """
        self.logger.debug(f"Converting JSON data to DataFrame for API: {api_source}")

        if api_source == 'alphavantage':
            return self._parse_alphavantage_data(data)
        elif api_source == 'polygon':
            return self._parse_polygon_data(data)
        elif api_source == 'yfinance':
            return self._parse_yfinance_data(data)
        elif api_source == 'finnhub_quote':
            return self._parse_finnhub_quote_data(data)
        elif api_source == 'finnhub_metrics':
            return self._parse_finnhub_metrics_data(data)
        elif api_source == 'finnhub_symbols':
            return self._parse_finnhub_symbols_data(data)
        elif api_source == 'company_news':
            return self._parse_company_news_data(data)
        elif api_source == 'alpaca':
            return self._parse_alpaca_data(data)
        else:
            self.logger.error(f"Unknown API source: {api_source}")
            raise ValueError(f"Unknown API source: {api_source}")

    # -------------------------------------------------------------------
    # Section 9: Data Parsing for Different APIs
    # -------------------------------------------------------------------
    
    def _parse_alphavantage_data(self, data: dict) -> pd.DataFrame:
        """Parses Alpha Vantage data into a pandas DataFrame."""
        time_series_key = "Time Series (Daily)"
        time_series = data.get(time_series_key, {})

        if not time_series:
            self.logger.error(f"No data found for key: {time_series_key}")
            raise ValueError(f"No data found for key: {time_series_key}")

        results = [
            {
                'date': date,
                'open': float(values["1. open"]),
                'high': float(values["2. high"]),
                'low': float(values["3. low"]),
                'close': float(values["4. close"]),
                'volume': int(values["5. volume"])
            }
            for date, values in time_series.items()
        ]

        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _parse_polygon_data(self, data: dict) -> pd.DataFrame:
        """Parses Polygon.io data into a pandas DataFrame."""
        results = data.get("results", [])

        if not results:
            self.logger.error("No data found in Polygon response.")
            raise ValueError("No data found in Polygon response.")

        parsed_results = [
            {
                'date': datetime.utcfromtimestamp(item['t'] / 1000).strftime('%Y-%m-%d'),
                'open': item['o'],
                'high': item['h'],
                'low': item['l'],
                'close': item['c'],
                'volume': item['v']
            }
            for item in results
        ]

        df = pd.DataFrame(parsed_results)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _parse_yfinance_data(self, data: dict) -> pd.DataFrame:
        """Parses Yahoo Finance data into a pandas DataFrame."""
        try:
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]

            df = pd.DataFrame({
                'date': pd.to_datetime(timestamps, unit='s'),
                'open': quote['open'],
                'high': quote['high'],
                'low': quote['low'],
                'close': quote['close'],
                'volume': quote['volume']
            })

            df.set_index('date', inplace=True)
            return df

        except (KeyError, IndexError) as e:
            self.logger.error(f"Error parsing Yahoo Finance data: {e}")
            raise ValueError("Invalid structure in Yahoo Finance response.")

    def _parse_finnhub_quote_data(self, data: dict) -> pd.DataFrame:
        """Parses Finnhub quote data into a pandas DataFrame."""
        required_keys = ['c', 'd', 'dp', 'h', 'l', 'o', 'pc', 't']
        if not all(key in data for key in required_keys):
            self.logger.error("Missing quote data in Finnhub response.")
            raise ValueError("Invalid structure in Finnhub quote data.")

        df = pd.DataFrame([{
            'date': datetime.utcfromtimestamp(data['t']),
            'current_price': data['c'],
            'change': data['d'],
            'percent_change': data['dp'],
            'high': data['h'],
            'low': data['l'],
            'open': data['o'],
            'previous_close': data['pc']
        }])

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _parse_finnhub_metrics_data(self, data: dict) -> pd.DataFrame:
        """Parses Finnhub financial metrics data into a pandas DataFrame."""
        metrics = data.get("metric", {})
        if not metrics:
            self.logger.error("No metric data found in Finnhub response.")
            raise ValueError("Invalid structure in Finnhub metrics data.")

        df = pd.DataFrame([metrics])
        df['date_fetched'] = datetime.utcnow()
        df.set_index('date_fetched', inplace=True)
        return df

    def _parse_finnhub_symbols_data(self, data: list) -> pd.DataFrame:
        """Parses Finnhub stock symbols data into a pandas DataFrame."""
        if not isinstance(data, list) or not data:
            self.logger.error("No symbol data found in Finnhub response.")
            raise ValueError("Invalid structure in Finnhub symbols data.")

        df = pd.DataFrame(data)
        return df

    def _parse_company_news_data(self, data: list) -> pd.DataFrame:
        """Parses Finnhub company news data into a pandas DataFrame."""
        if not isinstance(data, list):
            self.logger.error("Company news data is not in expected list format.")
            raise ValueError("Company news data is not in expected list format.")
        
        if not data:
            self.logger.warning("No company news found for the given date range.")
            return pd.DataFrame()  # Return empty DataFrame

        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df.set_index('datetime', inplace=True)
        return df

    def _parse_alpaca_data(self, data: dict) -> pd.DataFrame:
        """Parses Alpaca data into a pandas DataFrame."""
        bars = data.get('bars', [])

        if not bars:
            self.logger.error("No data found in Alpaca response.")
            raise ValueError("No data found in Alpaca response.")

        parsed_results = [
            {
                'date': datetime.strptime(bar['t'], '%Y-%m-%dT%H:%M:%SZ'),
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v']
            }
            for bar in bars
        ]

        df = pd.DataFrame(parsed_results)
        df.set_index('date', inplace=True)
        return df

    # -------------------------------------------------------------------
    # Section 10: Saving Data to CSV
    # -------------------------------------------------------------------
    
    @staticmethod
    def save_to_csv(data: pd.DataFrame, symbol: str, directory: Path, logger: logging.Logger, filename_suffix: str = "") -> None:
        """
        Saves the data to CSV in the specified directory.

        :param data: Pandas DataFrame containing the data.
        :param symbol: Stock symbol (e.g., 'AAPL').
        :param directory: Directory path where the CSV will be saved.
        :param logger: Logger instance for logging.
        :param filename_suffix: Optional suffix for the filename to distinguish data types.
        """
        if filename_suffix:
            filename = f"{symbol}_{filename_suffix}.csv"
        else:
            filename = f"{symbol}_data.csv"

        file_path = directory / filename
        data.to_csv(file_path, index=True)
        logger.info(f"Data for {symbol} saved to {file_path}")

# -------------------------------------------------------------------
# Example Usage (Main Section)
# -------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        fetcher = DataFetchUtils()

        # Initialize Alpaca API client
        try:
            alpaca_api = initialize_alpaca()
            fetcher.logger.info("Alpaca API client initialized successfully.")
        except EnvironmentError as e:
            fetcher.logger.error(f"Failed to initialize Alpaca API client: {e}")
            return  # Exit if Alpaca API cannot be initialized

        async with aiohttp.ClientSession() as session:
            # Construct headers for Alpaca API requests
            headers_alpaca = {
                "APCA-API-KEY-ID": os.getenv('ALPACA_API_KEY'),
                "APCA-API-SECRET-KEY": os.getenv('ALPACA_SECRET_KEY')
            }

            # Alpaca API URL
            alpaca_url = f"https://data.alpaca.markets/v2/stocks/AAPL/bars?timeframe=1Day"

            # Example API URLs
            api_key_alpha = os.getenv('ALPHAVANTAGE_API_KEY', 'dummy_key')
            api_key_polygon = os.getenv('POLYGONIO_API_KEY', 'dummy_key')
            api_key_finnhub = os.getenv('FINNHUB_API_KEY', 'dummy_key')
            # Note: YFinance typically doesn't require an API key
            yfinance_url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"

            # Finnhub endpoints
            finnhub_quote_url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key_finnhub}"
            finnhub_metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol=AAPL&metric=all&token={api_key_finnhub}"
            finnhub_symbols_url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={api_key_finnhub}"
            # Adjust the date range as needed for company news
            finnhub_news_url = f"https://finnhub.io/api/v1/company-news?symbol=AAPL&from=2024-01-01&to=2024-10-17&token={api_key_finnhub}"

            # Alpha Vantage and Polygon URLs
            alphavantage_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={api_key_alpha}"
            polygon_url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-12-31?adjusted=true&apiKey={api_key_polygon}"

            try:
                # 1. Fetch Alpaca Data
                alpaca_data = await fetcher.fetch_with_retries(alpaca_url, headers_alpaca, session)
                if alpaca_data:
                    try:
                        alpaca_df = fetcher.convert_json_to_dataframe(alpaca_data, "alpaca")
                        print("Alpaca DataFrame Head:")
                        print(alpaca_df.head())
                        # Define Alpaca data directory
                        alpaca_data_dir = base_data_dir / "Alpaca"
                        alpaca_data_dir.mkdir(parents=True, exist_ok=True)
                        fetcher.save_to_csv(alpaca_df, "AAPL", alpaca_data_dir, fetcher.logger)
                    except ValueError as ve:
                        fetcher.logger.error(f"Error processing Alpaca data: {ve}")

                # 2. Fetch Alpha Vantage Data
                alpha_data = await fetcher.fetch_with_retries(alphavantage_url, headers={}, session=session)
                if alpha_data:
                    try:
                        alpha_df = fetcher.convert_json_to_dataframe(alpha_data, "alphavantage")
                        print("Alpha Vantage DataFrame Head:")
                        print(alpha_df.head())
                        # Define Alpha Vantage data directory
                        alphavantage_data_dir = base_data_dir / "AlphaVantage"
                        alphavantage_data_dir.mkdir(parents=True, exist_ok=True)
                        fetcher.save_to_csv(alpha_df, "AAPL", alphavantage_data_dir, fetcher.logger)
                    except ValueError as ve:
                        fetcher.logger.error(f"Error processing Alpha Vantage data: {ve}")

                # 3. Fetch Polygon Data
                polygon_data = await fetcher.fetch_with_retries(polygon_url, headers={}, session=session)
                if polygon_data:
                    try:
                        polygon_df = fetcher.convert_json_to_dataframe(polygon_data, "polygon")
                        print("Polygon DataFrame Head:")
                        print(polygon_df.head())
                        # Define Polygon data directory
                        polygon_data_dir = base_data_dir / "Polygon"
                        polygon_data_dir.mkdir(parents=True, exist_ok=True)
                        fetcher.save_to_csv(polygon_df, "AAPL", polygon_data_dir, fetcher.logger)
                    except ValueError as ve:
                        fetcher.logger.error(f"Error processing Polygon data: {ve}")

                # 4. Fetch Yahoo Finance Data
                yfinance_data = await fetcher.fetch_with_retries(yfinance_url, headers={}, session=session)
                if yfinance_data:
                    try:
                        yfinance_df = fetcher.convert_json_to_dataframe(yfinance_data, "yfinance")
                        print("Yahoo Finance DataFrame Head:")
                        print(yfinance_df.head())
                        # Define Yahoo Finance data directory
                        yfinance_data_dir = base_data_dir / "YahooFinance"
                        yfinance_data_dir.mkdir(parents=True, exist_ok=True)
                        fetcher.save_to_csv(yfinance_df, "AAPL", yfinance_data_dir, fetcher.logger)
                    except ValueError as ve:
                        fetcher.logger.error(f"Error processing Yahoo Finance data: {ve}")

                # 5. Fetch Finnhub Quote Data
                finnhub_quote_data = await fetcher.fetch_finnhub_quote("AAPL", session)
                if finnhub_quote_data:
                    try:
                        finnhub_quote_df = fetcher.convert_json_to_dataframe(finnhub_quote_data, "finnhub_quote")
                        print("Finnhub Quote DataFrame:")
                        print(finnhub_quote_df)
                        # Define Finnhub Quotes data directory
                        finnhub_quotes_dir = base_data_dir / "Finnhub" / "Quotes"
                        finnhub_quotes_dir.mkdir(parents=True, exist_ok=True)
                        fetcher.save_to_csv(finnhub_quote_df, "AAPL_quote", finnhub_quotes_dir, fetcher.logger)
                    except ValueError as ve:
                        fetcher.logger.error(f"Error processing Finnhub Quote data: {ve}")

                # 6. Fetch Finnhub Financial Metrics
                finnhub_metrics_data = await fetcher.fetch_finnhub_metrics("AAPL", session)
                if finnhub_metrics_data:
                    try:
                        finnhub_metrics_df = fetcher.convert_json_to_dataframe(finnhub_metrics_data, "finnhub_metrics")
                        print("Finnhub Financial Metrics DataFrame Head:")
                        print(finnhub_metrics_df.head())
                        # Define Finnhub Metrics data directory
                        finnhub_metrics_dir = base_data_dir / "Finnhub" / "Metrics"
                        finnhub_metrics_dir.mkdir(parents=True, exist_ok=True)
                        fetcher.save_to_csv(finnhub_metrics_df, "AAPL_metrics", finnhub_metrics_dir, fetcher.logger)
                    except ValueError as ve:
                        fetcher.logger.error(f"Error processing Finnhub Metrics data: {ve}")

                # 7. Fetch Finnhub Stock Symbols
                finnhub_symbols_data = await fetcher.fetch_finnhub_symbols("US", session)
                if finnhub_symbols_data:
                    try:
                        finnhub_symbols_df = fetcher.convert_json_to_dataframe(finnhub_symbols_data, "finnhub_symbols")
                        print("Finnhub Stock Symbols DataFrame (First 5 Rows):")
                        print(finnhub_symbols_df.head())
                        # Define Finnhub Symbols data directory
                        finnhub_symbols_dir = base_data_dir / "Finnhub" / "Symbols"
                        finnhub_symbols_dir.mkdir(parents=True, exist_ok=True)
                        fetcher.save_to_csv(finnhub_symbols_df, "US_symbols", finnhub_symbols_dir, fetcher.logger)
                    except ValueError as ve:
                        fetcher.logger.error(f"Error processing Finnhub Symbols data: {ve}")

                # 8. Fetch Finnhub Company News
                finnhub_news_data = await fetcher.fetch_company_news("AAPL", "2024-01-01", "2024-10-17", session)
                if finnhub_news_data:
                    try:
                        finnhub_news_df = fetcher.convert_json_to_dataframe(finnhub_news_data, "company_news")
                        if not finnhub_news_df.empty:
                            print("Finnhub Company News DataFrame Head:")
                            print(finnhub_news_df.head())
                            # Define Finnhub News data directory
                            finnhub_news_dir = base_data_dir / "Finnhub" / "News"
                            finnhub_news_dir.mkdir(parents=True, exist_ok=True)
                            fetcher.save_to_csv(finnhub_news_df, "AAPL_news", finnhub_news_dir, fetcher.logger)
                        else:
                            print("No company news found for the given date range.")
                    except ValueError as ve:
                        fetcher.logger.error(f"Error processing Finnhub Company News data: {ve}")

            except Exception as e:
                fetcher.logger.error(f"An error occurred: {e}")

    # Run the main async function
    asyncio.run(main())
# -------------------------------------------------------------------
# File: C:/Projects/TradingRobotPlug/Scripts/Utilities/Analysis/data_fetching_utils.py
# Description: Utility functions for fetching stock and news data, with
#              support for asynchronous operations and improved error handling.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import logging
import os
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
from textblob import TextBlob
from requests.exceptions import HTTPError
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import aiohttp

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------
# Dynamically determine the project root and load environment variables
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Retrieve API key from environment variables
NEWS_API_KEY = os.getenv('NEWSAPI_API_KEY', 'your_default_api_key')

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# -------------------------------------------------------------------
# Stock Data Fetching Functions
# -------------------------------------------------------------------

def get_stock_data(ticker, start_date="2022-01-01", end_date=None, interval="1d"):
    """
    Fetches historical stock data for a given ticker using yfinance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for fetching data (YYYY-MM-DD).
        end_date (str): End date for fetching data (YYYY-MM-DD).
        interval (str): Data interval (e.g., '1d', '1h').

    Returns:
        pd.DataFrame: DataFrame containing stock data.
    """
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    try:
        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        # Ensure 'Date' is a column, not an index
        if 'Date' not in data.columns:
            data.reset_index(inplace=True)
        
        # Validate the presence of 'Date'
        if 'Date' not in data.columns:
            raise ValueError("Stock data does not contain 'Date' column after reset_index.")

        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data['symbol'] = ticker
        logger.info(f"Fetched {len(data)} records for ticker {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

# -------------------------------------------------------------------
# Asynchronous Fetching for Stock Data
# -------------------------------------------------------------------
async def fetch_stock_data_async(ticker, start_date="2022-01-01", end_date=None, interval="1d"):
    """
    Asynchronously fetches stock data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for fetching data (YYYY-MM-DD).
        end_date (str): End date for fetching data (YYYY-MM-DD).
        interval (str): Data interval (e.g., '1d').

    Returns:
        pd.DataFrame: DataFrame containing stock data.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_stock_data, ticker, start_date, end_date, interval)

# -------------------------------------------------------------------
# News Data Fetching Functions
# -------------------------------------------------------------------

def get_news_data(ticker, page_size=5):
    """
    Fetches news articles related to a stock ticker using News API.

    Args:
        ticker (str): Stock ticker symbol.
        page_size (int): Number of articles to fetch.

    Returns:
        pd.DataFrame: DataFrame containing news articles.
    """
    url = f'https://newsapi.org/v2/everything?q={ticker}&pageSize={page_size}&apiKey={NEWS_API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        news_data = [
            {
                'date': pd.to_datetime(article['publishedAt']).date(),
                'headline': article['title'],
                'content': article.get('description', '') or '',
                'symbol': ticker,
                'source': article.get('source', {}).get('name', ''),
                'url': article.get('url', ''),
                'sentiment': TextBlob(article.get('description', '') or '').sentiment.polarity
            }
            for article in articles
        ]
        logger.info(f"Fetched {len(news_data)} news articles for ticker {ticker}")
        return pd.DataFrame(news_data)
    except HTTPError as http_err:
        logger.error(f"HTTP error fetching news for {ticker}: {http_err}")
    except requests.RequestException as req_err:
        logger.error(f"Request exception fetching news data for {ticker}: {req_err}")
    except Exception as e:
        logger.error(f"Unexpected error fetching news data for {ticker}: {e}")
    return pd.DataFrame()

# -------------------------------------------------------------------
# Asynchronous Fetching for News Data
# -------------------------------------------------------------------
async def fetch_news_data_async(ticker, page_size=5):
    """
    Asynchronously fetches news articles for a given ticker.

    Args:
        ticker (str): Stock ticker symbol.
        page_size (int): Number of articles to fetch.

    Returns:
        pd.DataFrame: DataFrame containing news articles.
    """
    url = f'https://newsapi.org/v2/everything?q={ticker}&pageSize={page_size}&apiKey={NEWS_API_KEY}'
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                articles = data.get('articles', [])
                news_data = [
                    {
                        'date': pd.to_datetime(article['publishedAt']).date(),
                        'headline': article['title'],
                        'content': article.get('description', '') or '',
                        'symbol': ticker,
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'sentiment': TextBlob(article.get('description', '') or '').sentiment.polarity
                    }
                    for article in articles
                ]
                logger.info(f"Fetched {len(news_data)} news articles for ticker {ticker}")
                return pd.DataFrame(news_data)
        except aiohttp.ClientError as e:
            logger.error(f"Async HTTP error fetching news for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching news data asynchronously for {ticker}: {e}")
        return pd.DataFrame()

def save_to_csv(self, df: pd.DataFrame, symbol: str, directory: str = "./data"):
    """
    Saves the given DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be saved.
        symbol (str): Stock symbol, used to name the CSV file.
        directory (str): Directory where the CSV file will be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Define filename with symbol and current date
        filename = f"{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
        file_path = os.path.join(directory, filename)
        
        # Save DataFrame to CSV
        df.to_csv(file_path, index=False)
        self.logger.info(f"Data for {symbol} saved to CSV at {file_path}")
        
    except Exception as e:
        self.logger.error(f"Failed to save data for {symbol} to CSV: {e}", exc_info=True)
# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Fetch stock data for AAPL
    stock_data = get_stock_data("AAPL", start_date="2023-01-01")
    print(stock_data.head())

    # Example: Fetch news data for AAPL
    news_data = get_news_data("AAPL")
    print(news_data.head())

    # Example: Fetch data asynchronously
    async def main():
        stock_data_async = await fetch_stock_data_async("AAPL")
        news_data_async = await fetch_news_data_async("AAPL")
        print(stock_data_async.head())
        print(news_data_async.head())

    asyncio.run(main())

# -------------------------------------------------------------------
# Future Improvements
# -------------------------------------------------------------------
"""
1. **Implement Exponential Backoff for Retries**:
    - Add retry logic with exponential backoff for network requests.
    
2. **Enhanced Data Validation**:
    - Validate the fetched data to ensure completeness and quality.

3. **Multi-Ticker Support in Async Mode**:
    - Enhance asynchronous functions to handle multiple tickers in parallel.
    
4. **Rate Limiting for API Calls**:
    - Implement rate limiting handling to respect API limits.
    
5. **Logging Improvements**:
    - Add more granular logging for each step, including data shape and size.
"""
