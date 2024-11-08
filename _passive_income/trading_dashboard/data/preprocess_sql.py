# -------------------------------------------------------------------
# Filename: preprocess_sql.py
# Description: This script fetches daily adjusted stock data for a given symbol from Alpha Vantage,
#              processes it into a pandas DataFrame, and saves the raw data to a PostgreSQL database
#              using the DataStore class. It is fully configurable via environment variables
#              for portability across different environments.
# -------------------------------------------------------------------

import os
import sys
import requests
import pandas as pd
import logging
import time
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------------------
# Setup Project Root and PYTHONPATH
# -------------------------------------------------------------------

# Dynamically set the project root based on the current file's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Adjusted for the project structure
utilities_dir = project_root / 'Scripts' / 'Utilities'
data_processing_dir = project_root / 'Scripts' / 'Data_Processing'
model_dir = project_root / 'SavedModels'
model_utils = project_root / 'Scripts' / 'model_training' / 'utils'

# Extend sys.path once with all necessary directories
sys.path.extend([
    str(utilities_dir),
    str(data_processing_dir),
    str(model_dir),
    str(model_utils)
])

# -------------------------------------------------------------------
# Import ConfigManager and Logging Setup Using Absolute Imports
# -------------------------------------------------------------------
from config_handling.config_manager import ConfigManager
from config_handling.logging_setup import setup_logging
from data.data_store_interface import DataStoreInterface
from data.data_store import DataStore

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text  # Ensure 'text' is imported for raw SQL execution

# -------------------------------------------------------------------
# Preprocess SQL Script
# -------------------------------------------------------------------

def fetch_alpha_vantage_data(api_key: str, symbol: str, retries: int = 3, wait_time: int = 60) -> pd.DataFrame:
    """
    Fetch daily stock data from Alpha Vantage for the specified symbol using the free TIME_SERIES_DAILY endpoint.

    Args:
        api_key (str): Alpha Vantage API key.
        symbol (str): Stock symbol to fetch data for.
        retries (int): Number of retry attempts in case of rate limiting.
        wait_time (int): Time to wait between retries (in seconds).

    Returns:
        pd.DataFrame: DataFrame containing the stock data, or empty DataFrame if failed.
    """
    url = (
        f'https://www.alphavantage.co/query'
        f'?function=TIME_SERIES_DAILY'
        f'&symbol={symbol}'
        f'&outputsize=full'
        f'&apikey={api_key}'
    )

    for attempt in range(1, retries + 1):
        logging.info(f"Requesting data for {symbol} from Alpha Vantage. Attempt {attempt} of {retries}.")
        try:
            response = requests.get(url)
            logging.info(f"Response status code: {response.status_code}")
            logging.debug(f"Raw response content: {response.text}")

            if response.status_code == 200:
                data = response.json()
                if 'Time Series (Daily)' in data:
                    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                    df.reset_index(inplace=True)
                    df = df.rename(columns={
                        'index': 'Date',
                        '1. open': 'Open',
                        '2. high': 'High',
                        '3. low': 'Low',
                        '4. close': 'Close',
                        '5. volume': 'Volume'
                    })
                    df['Date'] = pd.to_datetime(df['Date'])
                    df['symbol'] = symbol  # Add the symbol column
                    df = df[['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    logging.info(f"Data for {symbol} fetched successfully.")
                    return df
                else:
                    logging.error(f"Expected 'Time Series (Daily)' not found in the response.")
                    logging.error(f"Full response: {data}")
                    break
            else:
                logging.error(f"Error fetching data from Alpha Vantage: {response.status_code}. Response content: {response.text}")
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception occurred: {e}")
            if attempt < retries:
                logging.info(f"Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logging.error("Maximum retry attempts reached due to request exceptions.")

    return pd.DataFrame()

def main():
    """
    Main function to fetch stock data and save it to the database.
    """
    # -------------------------------------------------------------------
    # Initialize ConfigManager and Logging
    # -------------------------------------------------------------------

    # Initialize ConfigManager
    config = ConfigManager(
        env_file=Path(".env"),
        required_keys=["ALPHAVANTAGE_API_KEY", "DATABASE_URL"]
    )

    # Setup logging
    log_dir = project_root / 'logs' / 'PreprocessSQL'
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        script_name="preprocess_sql",
        log_dir=log_dir,
        max_log_size=5 * 1024 * 1024,  # 5 MB
        backup_count=3,
        console_log_level=logging.INFO,
        file_log_level=logging.DEBUG,
        feedback_loop_enabled=True
    )

    # -------------------------------------------------------------------
    # Retrieve Configuration
    # -------------------------------------------------------------------
    api_key = config.get('ALPHAVANTAGE_API_KEY', required=True)
    symbol = config.get('STOCK_SYMBOL', default='TSLA')  # Default to 'TSLA' if not set
    db_uri = config.get('DATABASE_URL', required=True)  # e.g., 'postgresql://user:password@localhost:5432/tradingrobotplug'
    table_name = config.get('TABLE_NAME', default='trading_data')  # Default to 'trading_data' if not set

    # -------------------------------------------------------------------
    # Initialize DataStore
    # -------------------------------------------------------------------
    try:
        data_store = DataStore(config=config, logger=logger, use_csv=False)
        logger.info("DataStore initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DataStore: {e}")
        return

    # -------------------------------------------------------------------
    # Fetch and Save Data
    # -------------------------------------------------------------------
    raw_data = fetch_alpha_vantage_data(api_key, symbol)
    if not raw_data.empty:
        try:
            logger.info("Saving raw data to the database.")
            # Use DataStore's save_data_to_sql method for upsert
            data_store.save_data(raw_data, symbol, overwrite=True)
            logger.info("Data fetched and saved successfully using Alpha Vantage.")
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error occurred while saving data: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving data: {e}")
    else:
        logger.error("Failed to fetch data using Alpha Vantage.")

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------
# Future Improvements:
# - Add comprehensive error handling for all network and database operations.
# - Implement support for fetching data for multiple symbols in a single run.
# - Enhance logging to include more detailed information and timestamps.
# - Incorporate asynchronous requests to fetch data for multiple symbols concurrently.
# - Develop unit tests to ensure the reliability of each function.
# - Implement data validation and sanitization to ensure data integrity before saving.
# - Schedule the script to run at regular intervals using a scheduler like cron or Airflow.
# - Integrate with data visualization tools to provide insights from the fetched data.
# - Optimize data storage by handling incremental updates instead of replacing entire tables.
# - Consider adding support for additional data sources or APIs.
# -------------------------------------------------------------------
