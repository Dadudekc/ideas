import unittest
from unittest.mock import AsyncMock, patch
import pandas as pd
import logging
from pathlib import Path
import sys
from basicbot.data_fetcher import DataFetchUtils
from basicbot.config import DataFetchingConfig


class TestDataFetchUtils(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the DataFetchUtils class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up common resources for all test cases.
        """
        cls.logger = logging.getLogger("TestDataFetchUtils")
        logging.basicConfig(level=logging.INFO)

        cls.config = DataFetchingConfig(
            fetch_retries=2,
            backoff_factor=1
        )

    def setUp(self):
        """
        Set up a DataFetchUtils instance for each test.
        """
        self.data_fetcher = DataFetchUtils(logger=self.logger, config=self.config)

    @patch("yfinance.download")
    async def test_fetch_stock_data_success(self, mock_download):
        """
        Test successful data fetching.
        """
        # Mock yfinance data
        mock_data = pd.DataFrame({
            "Date": ["2023-11-15", "2023-11-16"],
            "Open": [100, 105],
            "High": [110, 115],
            "Low": [95, 100],
            "Close": [108, 110],
            "Volume": [1000, 1200],
        })
        mock_download.return_value = mock_data

        result = await self.data_fetcher.fetch_stock_data_async(
            ticker="AAPL", start_date="2023-11-01", end_date="2023-11-10", interval="1d"
        )

        self.assertFalse(result.empty)
        self.assertIn("symbol", result.columns)
        self.assertEqual(result.iloc[0]["symbol"], "AAPL")
        self.logger.info("Test fetch_stock_data_success passed.")

    @patch("yfinance.download")
    async def test_fetch_stock_data_empty(self, mock_download):
        """
        Test handling of empty data.
        """
        mock_download.return_value = pd.DataFrame()

        result = await self.data_fetcher.fetch_stock_data_async(
            ticker="INVALID", start_date="2023-11-01", end_date="2023-11-10", interval="1d"
        )

        self.assertTrue(result.empty)
        self.logger.info("Test fetch_stock_data_empty passed.")

    @patch("yfinance.download")
    async def test_fetch_stock_data_retries(self, mock_download):
        """
        Test retry mechanism for data fetching.
        """
        mock_download.side_effect = Exception("Simulated fetch error.")

        result = await self.data_fetcher.fetch_stock_data_async(
            ticker="AAPL", start_date="2023-11-01", end_date="2023-11-10", interval="1d"
        )

        self.assertTrue(result.empty)
        self.assertEqual(mock_download.call_count, self.config.fetch_retries + 1)
        self.logger.info("Test fetch_stock_data_retries passed.")

    def test_normalize_columns(self):
        """
        Test column normalization.
        """
        data = pd.DataFrame({
            "Open Price": [100],
            "Close Price": [105],
            "Trading Volume": [1000]
        })
        normalized_data = self.data_fetcher._normalize_columns(data)

        expected_columns = ["open_price", "close_price", "trading_volume"]
        self.assertEqual(list(normalized_data.columns), expected_columns)
        self.logger.info("Test normalize_columns passed.")

    async def test_fetch_stock_data_invalid_interval(self):
        """
        Test handling of invalid interval values.
        """
        with patch("yfinance.download", side_effect=ValueError("Invalid interval")):
            result = await self.data_fetcher.fetch_stock_data_async(
                ticker="AAPL", start_date="2023-11-01", end_date="2023-11-10", interval="10m"
            )
            self.assertTrue(result.empty)
            self.logger.info("Test fetch_stock_data_invalid_interval passed.")


if __name__ == "__main__":
    unittest.main()
