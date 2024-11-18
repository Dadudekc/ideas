import unittest
import pandas as pd
import logging
from unittest.mock import Mock
from basicbot.backtester import Backtester


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
        logging.basicConfig(level=logging.INFO)

    def setUp(self):
        """
        Set up a Backtester instance with a mock strategy and mocked dependencies.
        """
        self.mock_strategy = Mock()
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
            "close": [100, 102, 101, 104, 106],
        }
        df = pd.DataFrame(data)

        # Mock strategy behavior
        self.mock_strategy.calculate_indicators.return_value = df.assign(
            ema=[100, 101, 101, 102, 103],
            rsi=[30, 35, 40, 50, 55]
        )
        self.mock_strategy.evaluate.return_value = ['HOLD', 'BUY', 'HOLD', 'BUY', 'SELL']

        # Run backtest
        result = self.backtester.run_backtest(df)

        # Assertions
        self.assertIn("signal", result.columns)
        self.assertIn("position", result.columns)
        self.assertIn("returns", result.columns)
        self.assertIn("strategy_returns", result.columns)
        self.assertIn("cumulative_returns", result.columns)

        # Check specific values
        self.assertEqual(result['signal'].tolist(), ['HOLD', 'BUY', 'HOLD', 'BUY', 'SELL'])
        self.assertEqual(result['position'].tolist(), [0, 1, 1, 1, -1])  # Forward-filled positions
        
        # Adjusted precision for cumulative_returns
        self.assertAlmostEqual(result['cumulative_returns'].iloc[-1], 1.0392, places=4)  # Match calculated value

    def test_empty_dataframe(self):
        """
        Test the backtester with an empty DataFrame.
        """
        df = pd.DataFrame(columns=["close"])

        # Mock strategy behavior for empty DataFrame
        self.mock_strategy.calculate_indicators.return_value = pd.DataFrame(columns=["close"])
        self.mock_strategy.evaluate.return_value = []

        result = self.backtester.run_backtest(df)

        # Assertions
        self.assertTrue(result.empty)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
