import asyncio
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime
from typing import Optional
from basicbot.config import DataFetchingConfig


class DataFetchUtils:
    def __init__(self, logger: logging.Logger, config: DataFetchingConfig):
        """
        Initialize the DataFetchUtils class.

        Parameters:
        - logger: Logger instance for logging messages.
        - config: DataFetchingConfig instance containing configuration parameters.
        """
        self.logger = logger
        self.fetch_retries = config.fetch_retries
        self.backoff_factor = config.backoff_factor

    async def fetch_stock_data_async(
        self, ticker: str, start_date: str, end_date: str, interval: str
    ) -> pd.DataFrame:
        """
        Asynchronously fetch stock data using yfinance with retries.

        Parameters:
        - ticker: Stock ticker symbol.
        - start_date: Start date for fetching data (YYYY-MM-DD).
        - end_date: End date for fetching data (YYYY-MM-DD).
        - interval: Data interval (e.g., '1d', '1h').

        Returns:
        - DataFrame containing stock data, or an empty DataFrame if fetching fails.
        """
        attempt = 0
        while attempt <= self.fetch_retries:
            try:
                self.logger.info(
                    f"Fetching stock data for {ticker} from {start_date} to {end_date} at {interval} intervals. Attempt {attempt + 1}."
                )
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

                if data.empty:
                    raise ValueError(f"No data fetched for {ticker}. Ensure the ticker, dates, and interval are valid.")

                data.reset_index(inplace=True)
                data = self._normalize_columns(data)
                data['symbol'] = ticker
                return data

            except ValueError as ve:
                self.logger.error(f"ValueError: {ve}")
                return pd.DataFrame()

            except Exception as e:
                self.logger.error(
                    f"Error fetching data for {ticker}: {e}. Retrying in {self.backoff_factor ** attempt} seconds."
                )
                await asyncio.sleep(self.backoff_factor ** attempt)
                attempt += 1

        self.logger.error(f"Failed to fetch data for {ticker} after {self.fetch_retries} attempts.")
        return pd.DataFrame()

    @staticmethod
    def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names by converting to lowercase and replacing spaces with underscores.

        Parameters:
        - data: DataFrame to normalize.

        Returns:
        - Normalized DataFrame.
        """
        data.columns = [col.lower().replace(" ", "_") for col in data.columns]
        return data
