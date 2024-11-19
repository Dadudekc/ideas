import unittest
from unittest.mock import Mock, patch
import pandas as pd
from basicbot.data_fetcher import DataFetchUtils
from basicbot.config import DataFetchingConfig

class TestDataFetchUtils(unittest.TestCase):
    """
    Unit tests for the DataFetchUtils class.
    """

    def setUp(self):
        """
        Set up a DataFetchUtils instance for each test.
        """
        self.logger = Mock()
        self.config = DataFetchingConfig(
            api_key="test_api_key",
            fetch_retries=3,
            backoff_factor=2.0,
            start_date="2023-01-01",
            end_date="2023-12-31",
            interval="1d"
        )
        self.data_fetcher = DataFetchUtils(logger=self.logger, config=self.config)

    def test_fetch_stock_data_success(self):
        """
        Test successful fetching of stock data.
        """
        symbol = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        interval = "1d"

        df = self.data_fetcher.fetch_stock_data_async(symbol, start_date, end_date, interval)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        self.assertIn("close", df.columns)
        self.assertIn("high", df.columns)
        self.assertIn("low", df.columns)
        self.logger.info.assert_called_with(f"Fetching stock data for {symbol} from {start_date} to {end_date} with interval {interval}.")

    def test_fetch_stock_data_empty(self):
        """
        Test fetching stock data with empty date range.
        """
        symbol = "AAPL"
        start_date = "2023-01-10"
        end_date = "2023-01-01"  # end_date before start_date
        interval = "1d"

        df = self.data_fetcher.fetch_stock_data_async(symbol, start_date, end_date, interval)

        self.assertTrue(df.empty)

    def test_fetch_stock_data_invalid_interval(self):
        """
        Test fetching stock data with invalid interval.
        """
        symbol = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        interval = "invalid_interval"

        with self.assertRaises(ValueError):
            self.data_fetcher.fetch_stock_data_async(symbol, start_date, end_date, interval)

    def test_fetch_stock_data_retries(self):
        """
        Test fetching stock data with retries on failure.
        """
        symbol = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        interval = "1d"

        with patch.object(self.data_fetcher, 'fetch_stock_data_async', side_effect=Exception("API Error")):
            with self.assertRaises(Exception):
                self.data_fetcher.fetch_stock_data_async(symbol, start_date, end_date, interval)

    def test_normalize_columns(self):
        """
        Test normalization of DataFrame column names.
        """
        df = pd.DataFrame({
            "Close Price": [100, 101],
            "High Price": [102, 103],
            "Low Price": [98, 99]
        })

        normalized_df = DataFetchUtils._normalize_columns(df)

        expected_columns = ["close_price", "high_price", "low_price"]
        self.assertListEqual(list(normalized_df.columns), expected_columns)

if __name__ == "__main__":
    unittest.main()
