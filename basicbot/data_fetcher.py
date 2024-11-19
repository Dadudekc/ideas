import pandas as pd
import logging
from typing import Any, Dict

class DataFetchUtils:
    def __init__(self, logger: Any, config: Dict[str, Any]):
        """
        Initialize DataFetchUtils with logger and configuration.

        Parameters:
        - logger: Logger instance.
        - config: Configuration object containing necessary settings.
        """
        self.logger = logger
        self.config = config

    def fetch_stock_data_async(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Fetch stock data asynchronously.

        Parameters:
        - symbol: Stock symbol to fetch data for.
        - start_date: Start date for data fetching.
        - end_date: End date for data fetching.
        - interval: Data interval.

        Returns:
        - DataFrame containing stock data.
        """
        self.logger.info(f"Fetching stock data for {symbol} from {start_date} to {end_date} with interval {interval}.")
        # Implement real data fetching logic here, e.g., using yfinance or another API
        # For demonstration, returning mock data
        data = {
            "date": pd.date_range(start=start_date, end=end_date, freq=interval),
            "close": [100 + i for i in range((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)],
            "high": [101 + i for i in range((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)],
            "low": [99 + i for i in range((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1)],
        }
        df = pd.DataFrame(data)
        return df

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
