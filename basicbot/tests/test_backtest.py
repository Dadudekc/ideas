import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
import logging
import pandas as pd
import numpy as np
from basicbot.backtester import Backtester

# Add the project root to sys.path if necessary
project_root = Path(__file__).resolve().parent.parent.parent  # Adjust as needed
sys.path.insert(0, str(project_root))


class TestGenerateSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger("TestBacktester")
        if not cls.logger.handlers:
            logging.basicConfig(level=logging.CRITICAL)

    def setUp(self):
        mock_strategy = Mock()
        self.backtester = Backtester(strategy=mock_strategy, logger=self.logger)
        self.mock_strategy = self.backtester.strategy

    def test_buy_signal(self):
        """Test that BUY signal transitions to LONG position."""
        df = pd.DataFrame({"signal": ["BUY", "BUY", "HOLD"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [1, 1, 1]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_sell_signal(self):
        """Test that SELL signal transitions correctly."""
        df = pd.DataFrame({"signal": ["BUY", "SELL", "HOLD", "SELL"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [1, 0, 0, -1]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_hold_signal(self):
        """Test that HOLD signal maintains the current position."""
        df = pd.DataFrame({"signal": ["BUY", "HOLD", "HOLD", "SELL"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [1, 1, 1, 0]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_invalid_signal(self):
        """Test that invalid signals do not change the current position."""
        df = pd.DataFrame({"signal": ["BUY", "INVALID", "SELL", "UNKNOWN"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [1, 1, 0, -1]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        df = pd.DataFrame(columns=["signal"])
        self.mock_strategy.evaluate = Mock(return_value=[])

        result = self.backtester._generate_signals(df)
        self.assertTrue(result.empty, "Result DataFrame should be empty for empty input.")

    def test_large_dataset(self):
        """Test with a large dataset to ensure scalability."""
        df = pd.DataFrame({"signal": ["BUY"] + ["HOLD"] * 9999 + ["SELL"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [1] * 10000 + [0]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_mixed_signals(self):
        """Test with a mix of valid and invalid signals."""
        df = pd.DataFrame({"signal": ["BUY", "INVALID", "SELL", "HOLD", "UNKNOWN", "SELL"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [1, 1, 0, -1, -1, -1]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_sell_signal_enters_short(self):
        """Test that a SELL signal at position=0 enters SHORT."""
        df = pd.DataFrame({"signal": ["SELL", "HOLD"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [-1, -1]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_buy_signal_reverses_short_to_long(self):
        """Test that a BUY signal reverses from SHORT to LONG."""
        df = pd.DataFrame({"signal": ["SELL", "BUY", "HOLD"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [-1, 1, 1]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_repeated_signals_corrected(self):
        """Test with repeated BUY or SELL signals."""
        df = pd.DataFrame({"signal": ["BUY", "BUY", "SELL", "SELL", "HOLD"]})
        self.mock_strategy.evaluate = Mock(return_value=df['signal'].tolist())

        result = self.backtester._generate_signals(df)
        expected_positions = [1, 1, 0, -1, -1]  # Correctly transition to SHORT after second SELL
        self.assertEqual(result['position'].tolist(), expected_positions)


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
    Enhanced unit tests for the Backtester class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up common resources for all test cases.
        """
        cls.logger = logging.getLogger("TestBacktester")
        if not cls.logger.handlers:
            logging.basicConfig(level=logging.CRITICAL)  # Suppress debug logs during testing

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
        expected_columns = ['high', 'low', 'close', 'tr', 'atr', 'signal', 'position', 'returns', 'strategy_returns', 'cumulative_returns']
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")

        # Check specific values
        self.assertEqual(result['signal'].tolist(), ['HOLD', 'BUY', 'HOLD', 'BUY', 'SELL'])
        self.assertEqual(result['position'].tolist(), [0, 1, 1, 1, 0])  # Positions should be [0,1,1,1,0]

        # Check that cumulative_returns is calculated correctly
        expected_cumulative = [1.0, 1.0, 1.3, 1.5, 1.4]  # Adjusted based on strategy_returns
        np.testing.assert_almost_equal(result['cumulative_returns'].tolist(), expected_cumulative, decimal=6)

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
        self.assertTrue(result.empty, "Result DataFrame should be empty for empty input.")
        expected_columns = ["signal", "position", "tr", "atr", "returns", "strategy_returns", "cumulative_returns"]
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")

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

    def test_missing_columns(self):
        """
        Test the backtester raises an error when required columns are missing.
        """
        df = pd.DataFrame({"close": [1, 2, 3]})  # Missing 'high' and 'low'
        with self.assertRaises(ValueError):
            self.backtester._calculate_indicators(df)

    def test_generate_signals_positions(self):
        """
        Test signal generation and position tracking.
        """
        df = pd.DataFrame({
            "high": [10, 12, 15],
            "low": [8, 10, 11],
            "close": [9, 11, 13]
        })
        # Mock signals for evaluation
        self.mock_strategy.evaluate = Mock(return_value=['BUY', 'HOLD', 'SELL'])

        result = self.backtester._generate_signals(df)

        # Assertions for positions
        expected_positions = [1, 1, 0]  # Enter LONG, Maintain LONG, Exit LONG
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_position_maintenance(self):
        """
        Test that positions are maintained correctly for HOLD signals.
        """
        df = pd.DataFrame({
            "high": [20, 25, 15, 30],
            "low": [15, 18, 10, 25],
            "close": [18, 20, 12, 28]
        })
        self.mock_strategy.evaluate = Mock(return_value=['BUY', 'HOLD', 'HOLD', 'SELL'])

        result = self.backtester._generate_signals(df)

        # Validate position tracking
        expected_positions = [1, 1, 1, 0]  # Enter LONG, maintain LONG, maintain LONG, Exit LONG
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_calculate_returns(self):
        """
        Test return calculations for strategy and cumulative returns.
        """
        df = pd.DataFrame({
            "close": [100, 105, 110, 95],
            "signal": ['HOLD', 'BUY', 'HOLD', 'SELL'],
            "position": [0, 1, 1, 0]
        })
        result = self.backtester._calculate_returns(df)

        # Expected returns
        expected_returns = [0.0, 0.05, 0.047619, -0.136364]  # (close_t+1 - close_t) / close_t
        expected_strategy_returns = [0.0, 0.0, 0.047619, -0.136364]  # Strategy only active for positions
        expected_cumulative = [1.0, 1.0, 1.047619, 0.904762]  # Recalculated values to match logic

        # Assertions
        np.testing.assert_almost_equal(result['returns'].tolist(), expected_returns, decimal=6)
        np.testing.assert_almost_equal(result['strategy_returns'].tolist(), expected_strategy_returns, decimal=6)
        np.testing.assert_almost_equal(result['cumulative_returns'].tolist(), expected_cumulative, decimal=5)  # Adjust precision

    def test_full_backtest_flow(self):
        """
        Test full backtest flow from indicator calculation to returns.
        """
        df = pd.DataFrame({
            "high": [10, 12, 15, 13],
            "low": [8, 10, 11, 10],
            "close": [9, 11, 13, 12]
        })
        self.mock_strategy.evaluate = Mock(return_value=['BUY', 'HOLD', 'HOLD', 'SELL'])

        result = self.backtester.run_backtest(df)

        # Assertions for full flow
        expected_columns = ['high', 'low', 'close', 'tr', 'atr', 'signal', 'position', 'returns', 'strategy_returns', 'cumulative_returns']
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")

        self.assertEqual(result['signal'].tolist(), ['BUY', 'HOLD', 'HOLD', 'SELL'])
        self.assertEqual(result['position'].tolist(), [1, 1, 1, 0])

    def test_invalid_signals(self):
        """
        Test that invalid signals do not break the backtest.
        """
        df = pd.DataFrame({
            "high": [10, 12, 15],
            "low": [8, 10, 11],
            "close": [9, 11, 13]
        })
        self.mock_strategy.evaluate = Mock(return_value=['INVALID', 'HOLD', 'SELL'])

        result = self.backtester._generate_signals(df)

        # Assertions
        # Given 'SELL' at position=0 should set position=-1
        expected_positions = [0, 0, -1]  # Adjusted expectation
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_empty_signals(self):
        """
        Test backtester with empty or zero-length signals.
        """
        df = pd.DataFrame({
            "high": [],
            "low": [],
            "close": []
        })
        self.mock_strategy.evaluate = Mock(return_value=[])

        result = self.backtester.run_backtest(df)

        # Assertions for empty result
        self.assertTrue(result.empty, "Result DataFrame should be empty for empty input.")
        expected_columns = ["signal", "position", "tr", "atr", "returns", "strategy_returns", "cumulative_returns"]
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_high_volume_data(self):
        """
        Test with a large DataFrame to ensure scalability.
        """
        np.random.seed(42)
        df = pd.DataFrame({
            "high": np.random.uniform(100, 200, 10000),
            "low": np.random.uniform(50, 99, 10000),
            "close": np.random.uniform(75, 150, 10000)
        })
        self.mock_strategy.evaluate = Mock(return_value=['HOLD'] * len(df))

        result = self.backtester.run_backtest(df)

        # Ensure the DataFrame is processed correctly
        self.assertEqual(len(result), 10000)
        expected_columns = ['high', 'low', 'close', 'tr', 'atr', 'signal', 'position', 'returns', 'strategy_returns', 'cumulative_returns']
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_invalid_signals_no_position_change(self):
        """
        Ensure that invalid signals do not change the position from current.
        """
        df = pd.DataFrame({
            "high": [10, 12, 15],
            "low": [8, 10, 11],
            "close": [9, 11, 13]
        })
        self.mock_strategy.evaluate = Mock(return_value=['INVALID', 'HOLD', 'INVALID'])

        result = self.backtester._generate_signals(df)

        # Assertions
        # Expected positions remain unchanged at 0
        expected_positions = [0, 0, 0]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_unknown_signals_only(self):
        """
        Test with only unknown signals to ensure positions remain unchanged.
        """
        df = pd.DataFrame({
            "high": [10, 12],
            "low": [8, 10],
            "close": [9, 11]
        })
        self.mock_strategy.evaluate = Mock(return_value=['UNKNOWN', 'INVALID'])

        result = self.backtester._generate_signals(df)

        # Assertions
        # Expected positions remain at 0
        expected_positions = [0, 0]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_sell_signal_enters_short_position(self):
        """
        Test that a 'SELL' signal at position=0 enters a SHORT position.
        """
        df = pd.DataFrame({
            "high": [10, 12],
            "low": [8, 10],
            "close": [9, 11]
        })
        self.mock_strategy.evaluate = Mock(return_value=['SELL', 'HOLD'])

        result = self.backtester._generate_signals(df)

        # Expected positions: [ -1, -1 ]
        expected_positions = [-1, -1]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_buy_signal_enters_long_position_from_short(self):
        """
        Test that a 'BUY' signal reverses position from SHORT to LONG.
        """
        df = pd.DataFrame({
            "high": [10, 12],
            "low": [8, 10],
            "close": [9, 11]
        })
        self.mock_strategy.evaluate = Mock(return_value=['SELL', 'BUY'])

        result = self.backtester._generate_signals(df)

        # Expected positions: [ -1, 1 ]
        expected_positions = [-1, 1]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_invalid_signals_handling(self):
        """
        Test that the backtester correctly handles invalid signals by maintaining the current position.
        """
        df = pd.DataFrame({
            "high": [10, 12, 15],
            "low": [8, 10, 11],
            "close": [9, 11, 13]
        })
        self.mock_strategy.evaluate = Mock(return_value=['INVALID', 'HOLD', 'UNKNOWN'])

        # Run backtest
        result = self.backtester._generate_signals(df)

        # Assertions
        self.assertEqual(result['position'].tolist(), [0, 0, 0], "Invalid signals should maintain no position.")
        self.assertIn('signal', result.columns, "Signal column should exist in the result.")
        self.assertIn('position', result.columns, "Position column should exist in the result.")

    def test_all_unknown_signals(self):
        """
        Test that the backtester maintains the current position for all unknown signals.
        """
        df = pd.DataFrame({
            "high": [12, 14],
            "low": [10, 11],
            "close": [11, 13]
        })
        self.mock_strategy.evaluate = Mock(return_value=['UNKNOWN', 'INVALID'])

        result = self.backtester._generate_signals(df)

        # Expected positions remain 0 as no valid signals are given
        expected_positions = [0, 0]
        self.assertEqual(result['position'].tolist(), expected_positions)

    def test_mixed_signals_handling(self):
        """
        Test that the backtester correctly handles a mix of valid and invalid signals.
        """
        df = pd.DataFrame({
            "high": [10, 15, 14, 16],
            "low": [8, 9, 11, 12],
            "close": [9, 10, 13, 15]
        })
        self.mock_strategy.evaluate = Mock(return_value=['BUY', 'INVALID', 'SELL', 'HOLD'])

        result = self.backtester._generate_signals(df)

        # Expected positions: Enter LONG, maintain LONG (invalid signal), exit LONG, HOLD
        expected_positions = [1, 1, 0, 0]
        self.assertEqual(result['position'].tolist(), expected_positions)


if __name__ == "__main__":
    unittest.main()
