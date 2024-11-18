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
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import yfinance as yf
from textblob import TextBlob
from aiohttp import ClientSession, ClientConnectionError, ContentTypeError
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from logging.handlers import RotatingFileHandler

# -------------------------------------------------------------------
# Section 1: Configuration Handling
# -------------------------------------------------------------------

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file. If the file does not exist, create a default configuration,
    save it to the file, and return it.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data loaded from the file.
    """
    # Default configuration
    default_config = {
        "mode": "backtest",
        "symbol": "TSLA",
        "timeframe": "1d",  # Updated as per user's latest config
        "limit": 1000,
        "strategy": {
            "vwap_session": "RTH",
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
        },
        "project_root": "C:/Projects/#TODO/ideas/_passive_income/ArmyOfRobots/My1strobot",
        "log_dir": "C:/Projects/#TODO/ideas/_passive_income/ArmyOfRobots/My1strobot/logs",
        "logging": {
            "log_level": "INFO",
            "log_file": "logs/robot.log",
            "max_log_size": 5242880,  # 5MB
            "backup_count": 2
        },
        "data_fetching": {
            "fetch_retries": 5,
            "backoff_factor": 2,
            "cache_strategy": "memory"
        },
        "email_notifications": {
            "smtp_username": "",  # To be set via environment variables
            "smtp_password": "",  # To be set via environment variables
            "smtp_server": "",    # To be set via environment variables
            "smtp_port": 587,     # Default SMTP port for TLS
            "recipients": []      # List of recipient emails
        },
        "data_sources": ["Yahoo Finance", "Finnhub", "NewsAPI"]  # Excluding Alpaca due to subscription issues
    }

    # Ensure the config directory exists
    config_path = Path(config_file)
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create default config file if it doesn't exist
    if not config_path.exists():
        try:
            with open(config_path, "w") as file:
                yaml.dump(default_config, file, default_flow_style=False)
            print(f"Default configuration created at {config_file}. Please review and update it as necessary.")
        except Exception as e:
            print(f"Error creating default configuration: {e}")
            raise

    # Load the configuration from the file
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # Validate that the configuration is a dictionary
        if not isinstance(config, dict):
            raise ValueError("Configuration file is malformed or empty.")

        # Load environment variables and override defaults
        load_dotenv()
        # Environment variables take precedence
        env_config = {key: os.getenv(key) for key in os.environ if key in [
            "ALPACA_KEY_PLACEHOLDER",
            "ALPACA_SECRET_PLACEHOLDER",
            "ALPACA_BASE_URL",
            "FINNHUB_API_KEY",
            "NEWSAPI_API_KEY",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SMTP_SERVER",
            "SMTP_PORT"
        ]}
        # Convert relevant environment variables to correct types
        if "SMTP_PORT" in env_config and env_config["SMTP_PORT"]:
            env_config["SMTP_PORT"] = int(env_config["SMTP_PORT"])
        config = {**default_config, **config, **env_config}

        return config

    except Exception as e:
        print(f"Error loading configuration file {config_file}: {e}")
        raise

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

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory where log files will be stored.
        log_file (str): Path to the log file.
        max_log_size (int): Maximum size of the log file in bytes before rotation.
        backup_count (int): Number of backup log files to keep.
        log_level (str): Logging level (e.g., INFO, DEBUG).

    Returns:
        logging.Logger: Configured logger.
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
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    def __init__(self, logger: logging.Logger, config: Dict[str, Any]):
        """
        Initialize the DataFetchUtils class.
        All configurations are loaded via environment variables and config.yaml.
        """
        self.logger = logger
        self.config = config
        self.news_api_key = os.getenv('NEWSAPI_API_KEY') or config.get('NEWSAPI_API_KEY')
        if not self.news_api_key:
            self.logger.error("NewsAPI key is not set in environment variables or config.")
            raise EnvironmentError("Missing NewsAPI key.")

        # Initialize Alpaca API client
        try:
            self.alpaca_api = self.initialize_alpaca()
            if self.alpaca_api:
                self.logger.info("Alpaca API client initialized successfully.")
            else:
                self.logger.warning("Alpaca API client could not be initialized.")
        except EnvironmentError as e:
            self.logger.error(f"Alpaca API initialization failed: {e}")
            self.alpaca_api = None  # Continue without Alpaca if initialization fails

        # Initialize Database Connection
        self.db_engine = self.initialize_database()

    def initialize_alpaca(self) -> Optional[tradeapi.REST]:
        """
        Initializes Alpaca API client using credentials from environment variables.

        :return: Initialized Alpaca API client or None if credentials are missing.
        """
        api_key = os.getenv('ALPACA_KEY_PLACEHOLDER') or self.config.get('ALPACA_KEY_PLACEHOLDER')
        secret_key = os.getenv('ALPACA_SECRET_PLACEHOLDER') or self.config.get('ALPACA_SECRET_PLACEHOLDER')
        base_url = os.getenv('ALPACA_BASE_URL') or self.config.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not all([api_key, secret_key]):
            self.logger.error("Alpaca API credentials are not fully set in the environment variables or config.")
            return None

        self.logger.info("Initializing Alpaca API with provided credentials.")
        try:
            return tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca API client: {e}")
            return None

    def initialize_database(self) -> Optional[Any]:
        """
        Initializes the database connection using SQLAlchemy.

        :return: SQLAlchemy engine or None if failed.
        """
        db_url = os.getenv('DATABASE_URL') or self.config.get('DATABASE_URL')
        if not db_url:
            self.logger.warning("Database URL not provided. Database functionalities will be disabled.")
            return None

        try:
            engine = create_engine(db_url)
            self.logger.info("Database connection established successfully.")
            return engine
        except SQLAlchemyError as e:
            self.logger.error(f"Error connecting to the database: {e}")
            return None

    async def fetch_with_retries(self, url: str, headers: Dict[str, str], session: ClientSession, retries: int = 3) -> Optional[Any]:
        """
        Fetches data from the provided URL with retries on failure.

        :param url: The API URL to fetch.
        :param headers: Headers to include in the request (e.g., API keys).
        :param session: An active ClientSession.
        :param retries: Number of retry attempts in case of failure.
        :return: Parsed JSON data as a dictionary or list, or None if failed.
        """
        backoff = self.config['data_fetching'].get('backoff_factor', 2)
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

            await asyncio.sleep(backoff ** attempt)  # Exponential backoff for retries

        self.logger.error(f"Failed to fetch data after {retries} attempts: {url}")
        return None

    async def fetch_stock_data_async(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        self.logger.debug("Fetching stock data using Yahoo Finance.")
        interval = interval.lower()
        supported_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if interval not in supported_intervals:
            self.logger.warning(f"Interval '{interval}' not supported by Yahoo Finance. Falling back to '1d'.")
            interval = "1d"

        try:
            # Log parameters for debugging
            self.logger.debug(f"Yahoo Finance fetch parameters: Ticker={ticker}, Start={start_date}, End={end_date}, Interval={interval}")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            self.logger.debug(f"Yahoo Finance API response:\n{data.head() if not data.empty else 'Empty DataFrame'}")

            if data.empty:
                raise ValueError(f"Yahoo Finance returned no data for ticker {ticker}.")

            # Rename columns consistently
            data.reset_index(inplace=True)
            data.rename(columns={'Adj Close': 'adj_close', 'Date': 'date'}, inplace=True)

            # Check for required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in yfinance data: {missing_columns}")

            # Add symbol column
            data['symbol'] = ticker

            # Return clean DataFrame
            return data

        except Exception as e:
            self.logger.error(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()

    async def fetch_news_data_async(self, ticker: str, page_size: int = 5) -> pd.DataFrame:
        """
        Asynchronously fetches news articles for a given ticker using NewsAPI.

        Args:
            ticker (str): Stock ticker symbol.
            page_size (int): Number of articles to fetch.

        Returns:
            pd.DataFrame: DataFrame containing news articles.
        """
        url = f'https://newsapi.org/v2/everything?q={ticker}&pageSize={page_size}&apiKey={self.news_api_key}'
        async with aiohttp.ClientSession() as session:
            data = await self.fetch_with_retries(url, headers={}, session=session)
            if data and 'articles' in data:
                articles = data.get('articles', [])
                news_data = [
                    {
                        'date': pd.to_datetime(article['publishedAt']),
                        'headline': article['title'],
                        'content': article.get('description', '') or '',
                        'symbol': ticker,
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'sentiment': TextBlob(article.get('description', '') or '').sentiment.polarity
                    }
                    for article in articles
                ]
                self.logger.info(f"Fetched {len(news_data)} news articles for ticker {ticker}")
                return pd.DataFrame(news_data)
            else:
                self.logger.error("No articles found or failed to fetch news data.")
                return pd.DataFrame()

    async def fetch_finnhub_quote(self, symbol: str, session: ClientSession) -> Optional[Dict[str, Any]]:
        """
        Fetches the latest stock quote data for a given symbol from Finnhub.

        :param symbol: Stock symbol (e.g., 'AAPL').
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a dictionary or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY') or self.config.get('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables or config.")
            return None

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub quote for symbol: {symbol}")

        return await self.fetch_with_retries(url, headers={}, session=session)

    async def fetch_finnhub_metrics(self, symbol: str, session: ClientSession) -> Optional[Dict[str, Any]]:
        """
        Fetches basic financial metrics for a given symbol from Finnhub.

        :param symbol: Stock symbol (e.g., 'AAPL').
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a dictionary or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY') or self.config.get('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables or config.")
            return None

        url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub financial metrics for symbol: {symbol}")

        return await self.fetch_with_retries(url, headers={}, session=session)

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
        elif api_source == 'company_news':
            return self._parse_company_news_data(data)
        elif api_source == 'alpaca':
            return self._parse_alpaca_data(data)
        elif api_source == 'newsapi':
            return self._parse_newsapi_data(data)
        else:
            self.logger.error(f"Unknown API source: {api_source}")
            raise ValueError(f"Unknown API source: {api_source}")

    def _parse_alphavantage_data(self, data: dict) -> pd.DataFrame:
        """Parses Alpha Vantage data into a pandas DataFrame."""
        time_series_key = "Time Series (Daily)"
        time_series = data.get(time_series_key, {})

        if not time_series:
            self.logger.error(f"No data found for key: {time_series_key}")
            return pd.DataFrame()

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
            return pd.DataFrame()

        parsed_results = [
            {
                'date': datetime.utcfromtimestamp(item['t'] / 1000),
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

    def _parse_yfinance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parses Yahoo Finance data into a pandas DataFrame."""
        if data.empty:
            self.logger.error("Yahoo Finance data is empty.")
            return pd.DataFrame()

        # Ensure required columns are present
        required_columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'symbol']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing columns in Yahoo Finance data: {missing_columns}")
            return pd.DataFrame()

        return data

    def _parse_finnhub_quote_data(self, data: dict) -> pd.DataFrame:
        """Parses Finnhub quote data into a pandas DataFrame."""
        required_keys = ['c', 'd', 'dp', 'h', 'l', 'o', 'pc', 't']
        if not all(key in data for key in required_keys):
            self.logger.error("Missing quote data in Finnhub response.")
            return pd.DataFrame()

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
            return pd.DataFrame()

        df = pd.DataFrame([metrics])
        df['date_fetched'] = datetime.utcnow()
        df.set_index('date_fetched', inplace=True)
        return df

    def _parse_company_news_data(self, data: list) -> pd.DataFrame:
        """Parses NewsAPI company news data into a pandas DataFrame."""
        if not isinstance(data, list):
            self.logger.error("Company news data is not in expected list format.")
            return pd.DataFrame()

        if not data:
            self.logger.warning("No company news found for the given date range.")
            return pd.DataFrame()  # Return empty DataFrame

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _parse_alpaca_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parses Alpaca data into a pandas DataFrame."""
        if data.empty:
            self.logger.warning("Received empty DataFrame from Alpaca.")
            return data

        # Ensure 'date' is in datetime format
        if 'date' not in data.columns:
            self.logger.error("Alpaca data missing 'date' column.")
            return pd.DataFrame()

        # Rename columns to match the required format if necessary
        if 'timestamp' in data.columns:
            data.rename(columns={'timestamp': 'date'}, inplace=True)

        # Ensure required columns are present
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing columns in Alpaca data: {missing_columns}")
            return pd.DataFrame()

        return data

    def _parse_newsapi_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parses NewsAPI data into a pandas DataFrame."""
        if data.empty:
            self.logger.error("NewsAPI data is empty.")
            return pd.DataFrame()

        # Ensure required columns are present
        required_columns = ['date', 'headline', 'content', 'symbol', 'source', 'url', 'sentiment']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing columns in NewsAPI data: {missing_columns}")
            return pd.DataFrame()

        return data

    async def fetch_data_for_symbol(
        self,
        symbol: str,
        data_sources: List[str] = ["Yahoo Finance", "Finnhub", "NewsAPI"],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches data for a single stock symbol from various data sources.

        :param symbol: Stock symbol to fetch data for.
        :param data_sources: List of data sources to fetch data from.
        :param start_date: Start date for data fetching.
        :param end_date: End date for data fetching.
        :param interval: Data interval.

        :return: Dictionary containing DataFrames from each data source.
        """
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        all_data = {}

        async with aiohttp.ClientSession() as session:
            if "Yahoo Finance" in data_sources:
                yfinance_data = await self.fetch_stock_data_async(symbol, start_date, end_date, interval)
                if not yfinance_data.empty:
                    all_data["Yahoo Finance"] = yfinance_data
                else:
                    self.logger.warning("Yahoo Finance returned empty DataFrame.")

            if "Alpaca" in data_sources and self.alpaca_api:
                try:
                    alpaca_data = await self.fetch_alpaca_data_async(symbol, start_date, end_date, interval)
                    if not alpaca_data.empty:
                        all_data["Alpaca"] = alpaca_data
                    else:
                        self.logger.warning(f"Alpaca returned empty DataFrame for symbol: {symbol}.")
                except Exception as e:
                    self.logger.error(f"Error fetching Alpaca data for {symbol}: {e}")

            if "Finnhub" in data_sources:
                finnhub_quote = await self.fetch_finnhub_quote(symbol, session)
                finnhub_metrics = await self.fetch_finnhub_metrics(symbol, session)
                if finnhub_quote:
                    finnhub_quote_df = self.convert_json_to_dataframe(finnhub_quote, "finnhub_quote")
                    if not finnhub_quote_df.empty:
                        all_data["Finnhub Quote"] = finnhub_quote_df
                if finnhub_metrics:
                    finnhub_metrics_df = self.convert_json_to_dataframe(finnhub_metrics, "finnhub_metrics")
                    if not finnhub_metrics_df.empty:
                        all_data["Finnhub Metrics"] = finnhub_metrics_df

            if "NewsAPI" in data_sources:
                news_data = await self.fetch_news_data_async(symbol)
                if not news_data.empty:
                    newsapi_df = self.convert_json_to_dataframe(news_data, "newsapi")
                    if not newsapi_df.empty:
                        all_data["NewsAPI"] = newsapi_df

        # Log the fetched data sources
        for source, df in all_data.items():
            self.logger.debug(f"Data source '{source}' contains {len(df)} records.")

        return all_data

# -------------------------------------------------------------------
# Section 4: Strategy Class
# -------------------------------------------------------------------

class Strategy:
    """
    Strategy class encapsulates the parameters and logic required for trading strategies.

    Attributes:
        vwap_session (str): Session type for VWAP calculation.
        ema_length (int): Length of EMA for calculation.
        rsi_length (int): Length for RSI calculation.
        rsi_overbought (int): RSI overbought threshold.
        rsi_oversold (int): RSI oversold threshold.
        macd_fast (int): Fast EMA length for MACD calculation.
        macd_slow (int): Slow EMA length for MACD calculation.
        macd_signal (int): Signal line EMA length for MACD.
        adx_length (int): Length for ADX calculation.
        volume_threshold_length (int): Length for volume threshold calculation.
        atr_length (int): Length for ATR calculation.
        risk_percent (float): Percentage of portfolio to risk per trade.
        profit_target_percent (float): Profit target as a percentage.
        stop_multiplier (float): Stop-loss multiplier.
        trail_multiplier (float): Trailing stop multiplier.
        timeframe (str): Timeframe for strategy evaluation.
        limit (int): Limit for historical data fetching.
        required_length (int): Maximum length of data required for calculations.
        logger (Logger): Logger instance for the strategy class.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Strategy class with provided parameters.

        :param config: Dictionary containing strategy configuration.
        :param logger: Logger instance.
        """
        # Initialize strategy parameters with validation
        strategy_params = config.get("strategy", {})
        self.vwap_session = self._validate_string_param(
            strategy_params.get("vwap_session", "RTH"), "vwap_session"
        )
        self.ema_length = self._validate_numeric_param(
            strategy_params.get("ema_length", 8), "ema_length", min_value=1
        )
        self.rsi_length = self._validate_numeric_param(
            strategy_params.get("rsi_length", 14), "rsi_length", min_value=1
        )
        self.rsi_overbought = self._validate_numeric_param(
            strategy_params.get("rsi_overbought", 70), "rsi_overbought", min_value=0, max_value=100
        )
        self.rsi_oversold = self._validate_numeric_param(
            strategy_params.get("rsi_oversold", 30), "rsi_oversold", min_value=0, max_value=100
        )
        self.macd_fast = self._validate_numeric_param(
            strategy_params.get("macd_fast", 12), "macd_fast", min_value=1
        )
        self.macd_slow = self._validate_numeric_param(
            strategy_params.get("macd_slow", 26), "macd_slow", min_value=1
        )
        self.macd_signal = self._validate_numeric_param(
            strategy_params.get("macd_signal", 9), "macd_signal", min_value=1
        )
        self.adx_length = self._validate_numeric_param(
            strategy_params.get("adx_length", 14), "adx_length", min_value=1
        )
        self.volume_threshold_length = self._validate_numeric_param(
            strategy_params.get("volume_threshold_length", 20), "volume_threshold_length", min_value=1
        )
        self.atr_length = self._validate_numeric_param(
            strategy_params.get("atr_length", 14), "atr_length", min_value=1
        )
        self.risk_percent = self._validate_numeric_param(
            strategy_params.get("risk_percent", 1.0), "risk_percent", min_value=0, max_value=100
        )
        self.profit_target_percent = self._validate_numeric_param(
            strategy_params.get("profit_target_percent", 2.0), "profit_target_percent", min_value=0
        )
        self.stop_multiplier = self._validate_numeric_param(
            strategy_params.get("stop_multiplier", 1.5), "stop_multiplier", min_value=0
        )
        self.trail_multiplier = self._validate_numeric_param(
            strategy_params.get("trail_multiplier", 1.0), "trail_multiplier", min_value=0
        )
        self.timeframe = self._validate_string_param(
            config.get("timeframe", "5min"), "timeframe"
        )
        self.limit = self._validate_numeric_param(
            config.get("limit", 1000), "limit", min_value=1
        )

        # Calculate the required length for indicators
        self.required_length = max(
            self.ema_length,
            self.rsi_length,
            self.macd_slow,
            self.adx_length,
            self.volume_threshold_length,
            self.atr_length,
        ) + 1  # +1 for accurate calculations

        # Initialize logger
        self.logger = logger

    def _validate_numeric_param(self, value, name, min_value=None, max_value=None):
        """
        Validate a numeric parameter.

        :param value: Parameter value to validate.
        :param name: Name of the parameter.
        :param min_value: Minimum allowed value (inclusive).
        :param max_value: Maximum allowed value (inclusive).
        :return: Validated parameter value.
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Parameter '{name}' must be a numeric value. Got: {value}")
        if min_value is not None and value < min_value:
            raise ValueError(
                f"Parameter '{name}' must be at least {min_value}. Got: {value}"
            )
        if max_value is not None and value > max_value:
            raise ValueError(
                f"Parameter '{name}' must not exceed {max_value}. Got: {value}"
            )
        return value

    def _validate_string_param(self, value, name):
        """
        Validate a string parameter.

        :param value: Parameter value to validate.
        :param name: Name of the parameter.
        :return: Validated parameter value.
        """
        if not isinstance(value, str):
            raise ValueError(f"Parameter '{name}' must be a string. Got: {value}")
        return value

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading indicators on the provided DataFrame.

        :param df: DataFrame containing historical stock data.
        :return: DataFrame with calculated indicators or the original DataFrame if data is insufficient.
        """
        self.logger.debug("Starting indicator calculations.")

        # Validate the presence of required columns
        required_columns = self.config.get("required_columns", ["open", "high", "low", "close", "volume"])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing columns for indicator calculations: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        # Create a copy of the DataFrame to ensure no unintended mutations
        df_clean = df.copy()

        # Fill NaN values where possible to avoid unnecessary drops
        for col in required_columns:
            if df_clean[col].isnull().any():
                self.logger.warning(f"Column '{col}' contains NaN values. Filling with backfill method.")
                df_clean[col].fillna(method='bfill', inplace=True)

        # Ensure there is enough data for all required indicators
        if len(df_clean) < self.required_length:
            self.logger.warning(
                f"Not enough data to calculate indicators. Required: {self.required_length}, Available: {len(df_clean)}"
            )
            # Add NaN columns for indicators to avoid further errors
            indicator_columns = ['vwap', 'ema', 'rsi', 'macd_line', 'signal_line', 'macd_diff', 'adx']
            for col in indicator_columns:
                df_clean[col] = float('nan')
            return df_clean

        # Calculate indicators
        try:
            self.logger.debug(f"Calculating indicators using {len(df_clean)} rows of data.")
            self._calculate_vwap(df_clean)
            self._calculate_ema(df_clean)
            self._calculate_rsi(df_clean)
            self._calculate_macd(df_clean)
            self._calculate_adx(df_clean)
            self.logger.debug("All indicators calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            # Ensure indicator columns exist with NaN to maintain DataFrame consistency
            indicator_columns = ['vwap', 'ema', 'rsi', 'macd_line', 'signal_line', 'macd_diff', 'adx']
            for col in indicator_columns:
                if col not in df_clean.columns:
                    df_clean[col] = float('nan')
            return df_clean

        return df_clean

    def _calculate_vwap(self, df: pd.DataFrame):
        """Calculate Volume Weighted Average Price (VWAP)."""
        self.logger.debug("Calculating VWAP.")
        try:
            vwap_indicator = ta.volume.VolumeWeightedAveragePrice(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14  # VWAP typically uses a fixed window
            )
            df['vwap'] = vwap_indicator.volume_weighted_average_price()
            self.logger.debug("VWAP calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            raise

    def _calculate_ema(self, df: pd.DataFrame):
        """Calculate Exponential Moving Average (EMA)."""
        self.logger.debug("Calculating EMA.")
        try:
            ema_indicator = ta.trend.EMAIndicator(close=df['close'], window=self.ema_length)
            df['ema'] = ema_indicator.ema_indicator()
            self.logger.debug("EMA calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            raise

    def _calculate_rsi(self, df: pd.DataFrame):
        """Calculate Relative Strength Index (RSI)."""
        self.logger.debug("Calculating RSI.")
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_length)
            df['rsi'] = rsi_indicator.rsi()
            self.logger.debug("RSI calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            raise

    def _calculate_macd(self, df: pd.DataFrame):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        self.logger.debug("Calculating MACD.")
        try:
            macd = ta.trend.MACD(
                close=df['close'],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal
            )
            df['macd_line'] = macd.macd()
            df['signal_line'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            self.logger.debug("MACD calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            raise

    def _calculate_adx(self, df: pd.DataFrame):
        """Calculate the ADX indicator and add it to the DataFrame."""
        self.logger.debug("Calculating ADX.")
        try:
            # Ensure required columns are present
            required_columns = ["high", "low", "close"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' for ADX calculation.")

            # Strictly ensure there is enough data before calculation
            if len(df) <= self.adx_length:
                self.logger.warning(
                    f"Not enough data to calculate ADX. Required: {self.adx_length + 1}, Available: {len(df)}"
                )
                # Fill 'adx' column with NaN values
                df['adx'] = pd.Series([float('nan')] * len(df), index=df.index)
                return

            # Proceed with ADX calculation if data length is sufficient
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.adx_length
            )
            df['adx'] = adx_indicator.adx()
            self.logger.debug("ADX calculated successfully.")

        except IndexError as e:
            self.logger.error(f"IndexError during ADX calculation: {e}")
            # Ensure an 'adx' column exists to prevent further errors
            df['adx'] = pd.Series([float('nan')] * len(df), index=df.index)

        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}", exc_info=True)
            df['adx'] = pd.Series([float('nan')] * len(df), index=df.index)

    def evaluate(self, df: pd.DataFrame) -> List[str]:
        """
        Evaluate the latest indicators and generate trading signals.

        Parameters:
            df (pd.DataFrame): The DataFrame slice up to the current point in time.

        Returns:
            List[str]: A list of trading signals, e.g., ['BUY'] or ['SELL'].
        """
        self.logger.debug("Evaluating strategy based on indicators.")
        signals = []

        if df.empty:
            self.logger.debug("DataFrame slice is empty. No action taken.")
            return signals

        # Ensure required indicators are present in the DataFrame
        required_indicators = ['rsi', 'adx']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        if missing_indicators:
            self.logger.error(f"Missing indicators for evaluation: {missing_indicators}")
            return signals

        # Get the latest indicator values
        latest = df.iloc[-1]

        try:
            rsi_value = latest.get('rsi', None)
            adx_value = latest.get('adx', None)

            # Log indicator values for debugging
            self.logger.debug(f"Latest RSI: {rsi_value}, Latest ADX: {adx_value}")

            # Validate indicators before performing comparisons
            if pd.isnull(rsi_value) or pd.isnull(adx_value):
                self.logger.warning("Invalid RSI or ADX value detected. Skipping evaluation.")
                return signals

            # Example Strategy Logic:
            # Buy signal: RSI below oversold and ADX above a threshold (e.g., 25)
            if (rsi_value < self.rsi_oversold) and (adx_value > 25):
                signals.append('BUY')
                self.logger.debug("Buy signal generated.")

            # Sell signal: RSI above overbought and ADX above a threshold (e.g., 25)
            if (rsi_value > self.rsi_overbought) and (adx_value > 25):
                signals.append('SELL')
                self.logger.debug("Sell signal generated.")

        except Exception as e:
            self.logger.error(f"Error during signal evaluation: {e}")

        return signals

# -------------------------------------------------------------------
# Section 5: Email Notification Utility
# -------------------------------------------------------------------

class EmailNotifier:
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the EmailNotifier with SMTP configurations.

        :param config: Dictionary containing email configurations.
        :param logger: Logger instance.
        """
        email_config = config.get('email_notifications', {})
        self.smtp_username = os.getenv('SMTP_USERNAME') or email_config.get('smtp_username')
        self.smtp_password = os.getenv('SMTP_PASSWORD') or email_config.get('smtp_password')
        self.smtp_server = os.getenv('SMTP_SERVER') or email_config.get('smtp_server')
        self.smtp_port = int(os.getenv('SMTP_PORT') or email_config.get('smtp_port', 587))
        self.recipients = config.get('email_notifications', {}).get('recipients', [])
        self.logger = logger

        if not all([self.smtp_username, self.smtp_password, self.smtp_server, self.smtp_port, self.recipients]):
            self.logger.error("Incomplete SMTP configuration or no recipients. Email notifications will be disabled.")
            self.enabled = False
        else:
            self.enabled = True

    def send_email(self, subject: str, body: str, to_addresses: List[str]):
        """
        Sends an email with the given subject and body to the specified addresses.

        :param subject: Subject of the email.
        :param body: Body content of the email.
        :param to_addresses: List of recipient email addresses.
        """
        if not self.enabled:
            self.logger.warning("Email notifications are disabled due to incomplete configuration.")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = ", ".join(to_addresses)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            text = msg.as_string()
            server.sendmail(self.smtp_username, to_addresses, text)
            server.quit()
            self.logger.info(f"Email sent successfully to {to_addresses}")
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")

# -------------------------------------------------------------------
# Section 6: Main Execution Flow
# -------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame, required_columns: List[str], logger: logging.Logger, source: str) -> bool:
    """
    Validates if the DataFrame contains the required columns.

    :param df: DataFrame to validate.
    :param required_columns: List of required column names.
    :param logger: Logger instance.
    :param source: Name of the data source (for logging purposes).
    :return: True if DataFrame is valid, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns in {source} data: {missing_columns}")
        return False
    if df.empty:
        logger.warning(f"{source} returned an empty DataFrame.")
        return False
    return True

def fallback_to_sentiment(all_data: Dict[str, pd.DataFrame], logger: logging.Logger) -> List[str]:
    """
    Fallback strategy using news sentiment.

    :param all_data: Dictionary containing data from various sources.
    :param logger: Logger instance.
    :return: List of generated signals.
    """
    news_data = all_data.get("NewsAPI", pd.DataFrame())
    if not news_data.empty:
        avg_sentiment = news_data['sentiment'].mean()
        logger.info(f"Average sentiment score: {avg_sentiment:.2f}")

        if avg_sentiment > 0.2:
            signal = "BUY"
            logger.info("Sentiment suggests a BUY signal.")
        elif avg_sentiment < -0.2:
            signal = "SELL"
            logger.info("Sentiment suggests a SELL signal.")
        else:
            signal = "HOLD"
            logger.info("Sentiment is neutral. HOLD signal generated.")
        return [signal]
    else:
        logger.error("No fallback data (news sentiment) available.")
        return []

def main():
    # Setup and Initialization
    load_dotenv()
    try:
        config = load_config("config.yaml")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)

    logger = setup_logger("TradingRobot", config['log_dir'], **config['logging'])

    # Initialize utilities
    try:
        data_fetcher = DataFetchUtils(logger, config)
        email_notifier = EmailNotifier(config, logger)
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        sys.exit(1)

    # Configuration settings
    symbol = config.get("symbol", "TSLA")
    timeframe = config.get("timeframe", "1d")
    limit = config.get("limit", 30)
    required_columns = config.get("required_columns", ["open", "high", "low", "close", "volume"])

    # Validate timeframe
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    if timeframe not in valid_intervals:
        logger.error(f"Invalid interval '{timeframe}'. Must be one of {valid_intervals}.")
        sys.exit(1)

    # Calculate date range
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=limit)).strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"Error calculating date range: {e}")
        sys.exit(1)

    # Fetch data
    try:
        all_data = asyncio.run(
            data_fetcher.fetch_data_for_symbol(
                symbol=symbol,
                data_sources = config.get("data_sources", ["Yahoo Finance", "Finnhub", "NewsAPI"]),
                start_date=start_date,
                end_date=end_date,
                interval=timeframe
            )
        )
        logger.info(f"Data fetching completed for symbol: {symbol}")
    except Exception as e:
        logger.error(f"Error fetching data for symbol {symbol}: {e}")
        sys.exit(1)

    # Process and analyze data
    try:
        # Combine data sources and validate
        df = all_data.get("Yahoo Finance", pd.DataFrame())
        if df.empty or not validate_dataframe(df, required_columns, logger, "Yahoo Finance"):
            logger.warning("Yahoo Finance data is empty or invalid. Attempting to use Finnhub data.")
            # As Alpaca is causing subscription issues, we skip it for now
            # Instead, consider using Finnhub's quote or metrics as alternative
            df = all_data.get("Finnhub Quote", pd.DataFrame())
            if df.empty or not validate_dataframe(df, required_columns, logger, "Finnhub Quote"):
                logger.warning("Finnhub Quote data is also empty or invalid. Attempting fallback options.")
                signals = fallback_to_sentiment(all_data, logger)
                if signals:
                    logger.info(f"Generated signals from fallback: {signals}")
                    # Send email notifications about the signals
                    email_subject = f"Fallback Trading Signals for {symbol}"
                    email_body = (
                        f"Primary data sources failed. The following fallback trading signals have been generated "
                        f"for {symbol} based on news sentiment:\n" + ", ".join(signals)
                    )
                    recipient_emails = config.get("email_notifications", {}).get("recipients", [])
                    if recipient_emails:
                        email_notifier.send_email(email_subject, email_body, recipient_emails)
                    else:
                        logger.warning("No recipient emails configured.")
                else:
                    logger.info("No signals generated from fallback.")
                sys.exit(0)

        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"DataFrame size: {len(df)} rows")

        # Initialize and run strategy
        strategy = Strategy(config, logger)
        df_with_indicators = strategy.calculate_indicators(df)
        logger.info("Indicator calculations completed.")

        # Print and evaluate strategy
        print("\nDataFrame with indicators:")
        print(df_with_indicators.tail())

        signals = strategy.evaluate(df_with_indicators)
        if signals:
            logger.info(f"Generated signals: {signals}")
            print(f"Generated signals: {signals}")

            # Send email notifications
            email_subject = f"Trading Signals for {symbol}"
            email_body = (
                f"The following trading signals have been generated for {symbol}:\n" + ", ".join(signals)
            )
            recipient_emails = config.get("email_notifications", {}).get("recipients", [])
            if recipient_emails:
                email_notifier.send_email(email_subject, email_body, recipient_emails)
            else:
                logger.warning("No recipient emails configured.")
        else:
            logger.info("No trading signals generated.")
            print("No trading signals generated.")

    except Exception as e:
        logger.error(f"Error during strategy analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
