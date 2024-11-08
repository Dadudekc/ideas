# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/Scripts/Utilities/fetch_and_store_stock_data.py
# Description: Fetch stock data using yfinance, apply technical indicators,
#              and store it in PostgreSQL for analysis and training.
# -------------------------------------------------------------------

import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, List

import yfinance as yf
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, String, Float, Date, BigInteger, Integer, TIMESTAMP, Text
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
from dotenv import load_dotenv
import os
from logging.handlers import RotatingFileHandler

# -----------------------------
# Fallback Implementations
# -----------------------------

# 1. Basic Logging Setup
def setup_logging(log_name: str, log_dir: Path) -> logging.Logger:
    """
    Sets up basic logging with both console and file handlers.
    
    Args:
        log_name (str): Name of the logger.
        log_dir (Path): Directory where log files will be stored.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Ensure the logs directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3  # 5 MB per file
    )
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# 2. SQLAlchemy Base and `fetch_and_save_to_sql` Function
Base = declarative_base()

class AlphaVantageDaily(Base):
    __tablename__ = 'alpha_vantage_daily'
    symbol = Column(String(10), primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)

class NewsArticle(Base):
    __tablename__ = 'news_articles'
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    url = Column(Text, unique=True)
    published_at = Column(TIMESTAMP)
    source = Column(String(100))
    content = Column(Text)

def fetch_and_save_to_sql(session, ticker: str, df: pd.DataFrame, logger: logging.Logger, source: str = 'yfinance'):
    """
    Inserts stock data with technical indicators into PostgreSQL.
    
    Args:
        session: SQLAlchemy session object.
        ticker (str): Stock ticker symbol.
        df (pd.DataFrame): DataFrame containing stock data with indicators.
        logger (logging.Logger): Logger object for logging.
        source (str): Data source identifier.
    """
    if source == 'yfinance':
        for index, row in df.iterrows():
            record = AlphaVantageDaily(
                symbol=ticker,
                date=index.date(),
                open=row.get('Open'),
                high=row.get('High'),
                low=row.get('Low'),
                close=row.get('Close'),
                volume=row.get('Volume')
            )
            session.merge(record)  # Use merge to handle upserts
    elif source == 'newsapi':
        for article in df.to_dict(orient='records'):
            record = NewsArticle(
                title=article.get('title'),
                description=article.get('description'),
                url=article.get('url'),
                published_at=article.get('published_at'),
                source=article.get('source'),
                content=article.get('content')
            )
            session.merge(record)  # Use merge to handle upserts
    else:
        logger.warning(f"Unknown data source: {source}")
        return
    
    try:
        session.commit()
        logger.info(f"Inserted data for {ticker} from {source} successfully.")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to insert data for {ticker} from {source}: {e}")

# 3. Placeholder for `apply_all_indicators`
def apply_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all technical indicators to the stock data.
    Replace this function with your actual implementation.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
    
    Returns:
        pd.DataFrame: DataFrame with technical indicators added.
    """
    # Example: Adding simple moving averages (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    return df

# -----------------------------
# Main Script
# -----------------------------

# Load environment variables
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Debug print: Show the loaded environment variables related to database connection
print("Environment variables:")
print(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")
print(f"POSTGRES_DBNAME: {os.getenv('POSTGRES_DBNAME')}")
print(f"POSTGRES_USER: {os.getenv('POSTGRES_USER')}")
print(f"POSTGRES_PASSWORD: {os.getenv('POSTGRES_PASSWORD')}")
print(f"POSTGRES_PORT: {os.getenv('POSTGRES_PORT')}")

# Dynamically set the project root and other directories
script_dir = Path(__file__).resolve().parent
utilities_dir = project_root / 'Scripts' / 'Utilities'
data_processing_dir = project_root / 'Scripts' / 'Data_Processing'
model_dir = project_root / 'SavedModels'
model_utils = project_root / 'Scripts' / 'model_training' / 'utils'

# Debug print: Show the paths being added for imports
print(f"Adding to sys.path: {data_processing_dir}")
sys.path.append(str(data_processing_dir))

# Import necessary modules from the project structure
# Since modules are missing, we skip these imports
# Instead, we use the fallback implementations provided above

# -------------------------------------------------------------------
# Section 1: PostgreSQL connection setup via SQLAlchemy
# -------------------------------------------------------------------
def create_postgres_session() -> Optional[sessionmaker]:
    """
    Connect to the PostgreSQL database using SQLAlchemy and credentials from environment variables.
    
    Returns:
        SQLAlchemy sessionmaker instance if successful, None otherwise.
    """
    try:
        # Fetch database credentials from environment variables
        POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
        POSTGRES_DBNAME = os.getenv('POSTGRES_DBNAME', 'trading_robot_plug')
        POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
        POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
        POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5434')
    
        # Construct the PostgreSQL connection string
        DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DBNAME}"
        print(f"Connecting to PostgreSQL with DATABASE_URL: {DATABASE_URL}")
        
        engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
        
        # Create all tables if not already created
        Base.metadata.create_all(engine)
        
        # Return session maker
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

# -------------------------------------------------------------------
# Section 2: Fetch stock data with retry mechanism
# -------------------------------------------------------------------
def fetch_stock_data_with_retry(ticker: str, period: str = '1y', retries: int = 3, delay: int = 5) -> Optional[pd.DataFrame]:
    """
    Fetch stock data using yfinance with retry mechanism.
    
    Args:
        ticker (str): Stock ticker symbol.
        period (str): Data period (e.g., '1y').
        retries (int): Number of retry attempts.
        delay (int): Delay in seconds between retries.
    
    Returns:
        Optional[pd.DataFrame]: Fetched stock data or None if failed.
    """
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                raise ValueError(f"No data fetched for ticker {ticker}.")
            return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}. Retrying in {delay} seconds... Error: {e}")
            time.sleep(delay)
    print(f"Failed to fetch data for {ticker} after {retries} attempts.")
    return None

# -------------------------------------------------------------------
# Section 3: Validate OHLC Data
# -------------------------------------------------------------------
def validate_ohlc_data(df: pd.DataFrame, ticker: str) -> bool:
    """
    Check if the fetched data contains the required OHLC columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        ticker (str): Stock ticker symbol.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        print(f"Error processing {ticker}: DataFrame must contain 'Open', 'High', 'Low', and 'Close' columns")
        return False
    return True

# -------------------------------------------------------------------
# Section 4: Apply technical indicators to stock data
# -------------------------------------------------------------------
def apply_technical_indicators_to_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply technical indicators to the stock data.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
    
    Returns:
        pd.DataFrame: DataFrame with technical indicators added.
    """
    # Removed SQLite dependency for PostgreSQL exclusivity
    df_with_indicators = apply_all_indicators(df)
    return df_with_indicators

# -------------------------------------------------------------------
# Section 5: Insert stock data with indicators into PostgreSQL
# -------------------------------------------------------------------
def insert_data_with_indicators(session, ticker: str, df: pd.DataFrame, logger: logging.Logger):
    """
    Insert stock data with technical indicators into PostgreSQL.
    
    Args:
        session: SQLAlchemy session object.
        ticker (str): Stock ticker symbol.
        df (pd.DataFrame): DataFrame containing stock data with indicators.
        logger (logging.Logger): Logger object for logging.
    """
    df_with_indicators = apply_technical_indicators_to_data(df)
    fetch_and_save_to_sql(session, ticker, df_with_indicators, logger, source='yfinance')

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Setup logging
    logs_dir = project_root / 'logs'
    logger = setup_logging('fetch_and_store_stock_data', log_dir=logs_dir)

    # Connect to PostgreSQL
    session = create_postgres_session()
    if not session:
        logger.error("Failed to connect to the database.")
        print("Failed to connect to the database.")
        sys.exit(1)

    # Define tickers and fetch period
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    period = '10y'  # Fetch data for the past 10 years

    # Fetch, apply indicators, and store data for each ticker
    for ticker in tickers:
        try:
            logger.info(f"Fetching data for {ticker} using yfinance...")
            df = fetch_stock_data_with_retry(ticker, period)
            
            if df is not None and validate_ohlc_data(df, ticker):
                insert_data_with_indicators(session, ticker, df, logger)
                logger.info(f"Data for {ticker} inserted successfully.")
            else:
                logger.error(f"Failed to process data for {ticker} due to missing or incomplete data.")

        except ValueError as e:
            logger.error(f"Error processing {ticker}: {e}")
            print(f"Error processing {ticker}: {e}")

    # Close the SQLAlchemy session
    session.close()
    logger.info("Data fetching, indicator application, and insertion completed.")
    print("Data fetching, indicator application, and insertion completed.")

# -------------------------------------------------------------------
# Future Improvements:
#     - Add validation for performance metrics input to ensure consistent format.
#     - Implement a feature to update existing entries instead of always adding new records.
#     - Allow users to specify custom sorting criteria for data insertion.
#     - Optimize the technical indicators application for large datasets.
#     - Introduce versioning to track changes in technical indicators and data processing.
# -------------------------------------------------------------------
