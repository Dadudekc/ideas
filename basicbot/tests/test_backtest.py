import sys
from pathlib import Path
import unittest
import pandas as pd
import logging
from unittest.mock import Mock
from basicbot.backtester import Backtester

# Add the project root to sys.path if necessary
# Adjust the number of parent directories based on your project structure
# Here, assuming this test file is located at basicbot/tests/test_backtester.py
project_root = Path(__file__).resolve().parent.parent.parent  # Adjust as needed
sys.path.insert(0, str(project_root))


class MockStrategy:
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mock calculate_indicators method that does not modify the DataFrame.
        """
        return df

    def evaluate(self, df: pd.DataFrame) -> list:
        """
        Mock evaluate method that returns a list of signals based on DataFrame length.
        """
        return ['HOLD'] * len(df)


class TestBacktester(unittest.TestCase):
    """
    Unit tests for the Backtester class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up common resources for all test cases.
        """
        cls.logger = logging.getLogger("TestBacktester")
        if not cls.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def setUp(self):
        """
        Set up a Backtester instance with a mock strategy and mocked dependencies.
        """
        self.mock_strategy = MockStrategy()
        self.backtester = Backtester(
            strategy=self.mock_strategy,
            logger=self.logger,
            api=None,
            symbol="TEST",
            timeframe="1D",
            limit=100,
            portfolio=None,
            log_callback=None
        )

    def test_run_backtest(self):
        """
        Test the run_backtest method with mock strategy.
        """
        # Input data
        data = {
            "high": [10, 15, 14, 16, 17],
            "low": [8, 9, 11, 10, 12],
            "close": [9, 10, 13, 15, 14],
        }
        df = pd.DataFrame(data)

        # Mock strategy behavior for evaluate
        self.mock_strategy.evaluate = Mock(return_value=['HOLD', 'BUY', 'HOLD', 'BUY', 'SELL'])

        # Run backtest
        result = self.backtester.run_backtest(df)

        # Assertions
        self.assertIn("signal", result.columns)
        self.assertIn("position", result.columns)
        self.assertIn("returns", result.columns)
        self.assertIn("strategy_returns", result.columns)
        self.assertIn("cumulative_returns", result.columns)
        self.assertIn("atr", result.columns)

        # Check specific values
        self.assertEqual(result['signal'].tolist(), ['HOLD', 'BUY', 'HOLD', 'BUY', 'SELL'])
        self.assertEqual(result['position'].tolist(), [0, 1, 1, 1, 0])  # Positions should be [0,1,1,1,0]

        # Check that cumulative_returns is calculated
        self.assertFalse(result['cumulative_returns'].isnull().any(), "Cumulative returns should not contain NaN.")

    def test_empty_dataframe(self):
        """
        Test the backtester with an empty DataFrame.
        """
        # Ensure all necessary columns are present
        df = pd.DataFrame(columns=["high", "low", "close"])

        # Mock strategy behavior for empty DataFrame
        self.mock_strategy.evaluate = Mock(return_value=[])

        # Run the backtester
        result = self.backtester.run_backtest(df)

        # Assertions
        self.assertTrue(result.empty, "Result DataFrame should be empty for an empty input.")
        self.assertIn("atr", result.columns, "Result should contain the 'atr' column.")

    def test_atr_calculation(self):
        """
        Test ATR calculation within the backtester.
        """
        df = pd.DataFrame({
            "high": [10, 15, 14],
            "low": [8, 9, 11],
            "close": [9, 10, 13]
        })

        # Mock strategy behavior for evaluate
        self.mock_strategy.evaluate = Mock(return_value=['HOLD', 'HOLD', 'HOLD'])

        result = self.backtester._calculate_indicators(df)

        # Assertions
        self.assertIn("atr", result.columns, "ATR should be calculated and added to the DataFrame.")
        self.assertFalse(result['atr'].isnull().all(), "ATR should have some calculated values.")

        # Check that 'atr' is correctly calculated
        expected_tr = [
            max(10 - 8, abs(10 - 9), abs(8 - 9)),    # 2
            max(15 - 9, abs(15 - 10), abs(9 - 10)),  # 6
            max(14 - 11, abs(14 - 13), abs(11 - 13)) # 3
        ]
        expected_atr = pd.Series(expected_tr).rolling(window=14, min_periods=1).mean()
        pd.testing.assert_series_equal(result['atr'], expected_atr, check_names=False)


if __name__ == "__main__":
    unittest.main()
