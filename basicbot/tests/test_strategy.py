# test_strategy.py

import unittest
import pandas as pd
from unittest.mock import MagicMock
from strategy import Strategy, StrategyConfig, TradeSignal

class TestStrategy(unittest.TestCase):

    def setUp(self):
        """
        Set up mock data and a strategy instance for testing.
        """
        self.config = StrategyConfig()
        self.logger = MagicMock()
        self.strategy = Strategy(config=self.config, logger=self.logger)

        # Mock data with 60 rows to accommodate signals up to index 56
        self.df = pd.DataFrame({
            'close': [100 + i for i in range(60)],
            'high': [101 + i for i in range(60)],
            'low': [99 + i for i in range(60)],
            'volume': [1000 + (i * 10) for i in range(60)]
        })

    def test_calculate_indicators(self):
        """
        Test that the calculate_indicators method adds the expected columns to the DataFrame.
        """
        result = self.strategy.calculate_indicators(self.df.copy())

        # Ensure the DataFrame has enough rows for all indicators to calculate properly
        self.assertGreaterEqual(len(self.df), max(
            self.config.ema_length, self.config.rsi_length,
            self.config.macd_slow_window, self.config.bollinger_window,
            self.config.adx_window, self.config.atr_window
        ), "Insufficient rows in DataFrame for indicator calculation.")

        # Check for expected columns
        expected_columns = ['ema', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'adx', 'vwap', 'volume_sma_20', 'atr']
        for column in expected_columns:
            self.assertIn(column, result.columns, f"{column} should be in the DataFrame.")

    def test_calculate_indicators_insufficient_data(self):
        """
        Test that calculate_indicators raises an error when data is insufficient.
        """
        # Create DataFrame with insufficient rows
        insufficient_df = self.df.head(10)
        with self.assertRaises(ValueError, msg="Should raise ValueError for insufficient data rows"):
            self.strategy.calculate_indicators(insufficient_df)

    def test_calculate_indicators_missing_columns(self):
        """
        Test that calculate_indicators raises an error for missing required columns.
        """
        df_missing_columns = self.df.drop(columns=['volume'])
        with self.assertRaises(ValueError, msg="Should raise ValueError for missing columns"):
            self.strategy.calculate_indicators(df_missing_columns)

    def test_generate_signals_no_signals(self):
        """
        Test that generate_signals assigns HOLD when no conditions are met.
        """
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        # Override indicators to ensure no BUY or SELL conditions are met
        df_with_indicators['vwap'] = df_with_indicators['close'] - 10  # Ensures close > vwap
        df_with_indicators['ema'] = df_with_indicators['close'] + 10  # Ensures close < ema
        df_with_indicators['rsi'] = 50  # Between oversold and overbought
        df_with_indicators['macd_diff'] = 0
        df_with_indicators['adx'] = 10  # Below threshold
        df_with_indicators['volume'] = 1000  # Set volume below
        df_with_indicators['volume_sma_20'] = 1000
        result = self.strategy.generate_signals(df_with_indicators)

        # All signals should be HOLD
        self.assertTrue(all(result['signal'] == TradeSignal.HOLD), "All signals should be HOLD.")

    def test_generate_signals_buy_signal(self):
        """
        Test that generate_signals correctly assigns BUY signals.
        """
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        # Simulate BUY condition at index 14
        df_with_indicators.loc[14, 'close'] = df_with_indicators.loc[14, 'vwap'] + 1
        df_with_indicators.loc[14, 'ema'] = df_with_indicators.loc[14, 'close'] - 1
        df_with_indicators.loc[14, 'rsi'] = 35  # Above oversold threshold
        df_with_indicators.loc[14, 'macd_diff'] = 1.0
        df_with_indicators.loc[13, 'macd_diff'] = 0.0  # Crossover

        result = self.strategy.generate_signals(df_with_indicators)

        self.assertEqual(result.loc[14, 'signal'], TradeSignal.BUY, "Signal should be BUY at index 14.")

    def test_generate_signals_sell_signal(self):
        """
        Test that generate_signals correctly assigns SELL signals.
        """
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        # Simulate SELL condition at index 28
        df_with_indicators.loc[28, 'close'] = df_with_indicators.loc[28, 'vwap'] - 1
        df_with_indicators.loc[28, 'ema'] = df_with_indicators.loc[28, 'close'] + 1
        df_with_indicators.loc[28, 'rsi'] = 65  # Below overbought threshold
        df_with_indicators.loc[28, 'macd_diff'] = -1.0
        df_with_indicators.loc[27, 'macd_diff'] = 0.0  # Crossunder
        df_with_indicators.loc[28, 'adx'] = 25
        df_with_indicators.loc[28, 'volume'] = 2000
        df_with_indicators.loc[28, 'volume_sma_20'] = 1500

        result = self.strategy.generate_signals(df_with_indicators)

        self.assertEqual(result.loc[28, 'signal'], TradeSignal.SELL, "Signal should be SELL at index 28.")

    def test_backtest_strategy_no_trades(self):
        """
        Test backtest_strategy when no buy or sell signals are generated.
        """
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        # Override signals to HOLD
        df_with_indicators['signal'] = TradeSignal.HOLD
        result = self.strategy.backtest_strategy(df_with_indicators)

        # Positions should remain 0
        self.assertTrue(all(result['position'] == 0), "All positions should be 0 (flat).")
        # Equity should remain constant
        self.assertTrue(all(result['equity'] == 100000.0), "Equity should remain unchanged.")

    def test_backtest_strategy_immediate_stop_loss(self):
        """
        Test backtest_strategy when price immediately hits the stop loss.
        """
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())

        # Ensure ATR at index 14 is not NaN
        self.assertFalse(pd.isna(df_with_indicators.loc[14, 'atr']), "ATR at index 14 should not be NaN.")

        # Simulate a BUY signal at index 14 and immediate stop loss at index 15
        df_with_indicators['signal'] = TradeSignal.HOLD
        df_with_indicators.loc[14, 'signal'] = TradeSignal.BUY
        df_with_indicators.loc[14, 'close'] = 100.0  # Entry price
        df_with_indicators.loc[14, 'atr'] = 2.0  # Fixed ATR
        df_with_indicators.loc[15, 'close'] = 96.0  # Hits stop loss = 96.0
        df_with_indicators.loc[15, 'atr'] = 2.0  # Fixed ATR

        # Run backtest
        result = self.strategy.backtest_strategy(df_with_indicators)

        # Check that the position was entered and exited at index 14 and 15
        self.assertEqual(result.loc[14, 'position'], 1, "Position should be LONG at index 14.")
        self.assertEqual(result.loc[15, 'position'], 0, "Position should be exited at index 15.")
        expected_exit_price = 96.0
        self.assertAlmostEqual(result.loc[15, 'exit_price'], expected_exit_price, places=2,
                               msg="Exit price should match stop loss.")

    def test_backtest_strategy_multiple_trades(self):
        """
        Test backtest_strategy with multiple BUY and SELL signals.
        """
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())

        # Simulate multiple BUY and SELL signals
        signals = [TradeSignal.HOLD] * len(df_with_indicators)
        signals[14] = TradeSignal.BUY  # Enter LONG
        signals[28] = TradeSignal.SELL  # Exit LONG
        signals[42] = TradeSignal.BUY  # Enter LONG again
        signals[56] = TradeSignal.SELL  # Exit LONG again

        df_with_indicators['signal'] = signals

        # Set consistent ATR and close prices
        df_with_indicators.loc[[14, 28, 42, 56], 'atr'] = 2.0
        df_with_indicators.loc[14, 'close'] = 100.0  # Entry price for LONG
        df_with_indicators.loc[28, 'close'] = 96.0  # Exit LONG at Stop Loss
        df_with_indicators.loc[42, 'close'] = 100.0  # Entry price for LONG
        df_with_indicators.loc[56, 'close'] = 96.0  # Exit LONG at Stop Loss

        result = self.strategy.backtest_strategy(df_with_indicators)

        # Debugging output to validate intermediate data
        print(result[['signal', 'position', 'entry_price', 'exit_price', 'strategy_returns', 'equity']].iloc[[14, 28, 42, 56]])

        # Verify positions and resets
        self.assertEqual(result.loc[14, 'position'], 1, "Should be LONG at index 14.")
        self.assertEqual(result.loc[28, 'position'], 0, "Should exit LONG at index 28.")
        self.assertEqual(result.loc[42, 'position'], 1, "Should be LONG at index 42.")
        self.assertEqual(result.loc[56, 'position'], 0, "Should exit LONG at index 56.")

        # Additional checks for entry and exit prices
        self.assertEqual(result.loc[14, 'entry_price'], 100.0, "Entry price should be set at index 14.")
        self.assertEqual(result.loc[28, 'exit_price'], 96.0, "Exit price should match close price at index 28.")
        self.assertEqual(result.loc[42, 'entry_price'], 100.0, "Entry price should be set again at index 42.")
        self.assertEqual(result.loc[56, 'exit_price'], 96.0, "Exit price should match close price at index 56.")

        # Validate equity updates
        self.assertGreater(result.loc[28, 'equity'], 0, "Equity should update correctly after exit at index 28.")
        self.assertGreater(result.loc[56, 'equity'], 0, "Equity should update correctly after exit at index 56.")

    def test_logger_calls(self):
        """
        Test that logger.debug is called appropriately during strategy execution.
        """
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        df_with_signals = self.strategy.generate_signals(df_with_indicators)
        self.strategy.backtest_strategy(df_with_signals)

        # Each trade (enter and exit) should generate multiple debug logs
        # The exact number can vary based on the number of trades and conditions
        self.assertTrue(self.logger.debug.called, "Logger.debug should be called during strategy execution.")
        # Example: At least some debug calls are expected
        self.assertGreaterEqual(self.logger.debug.call_count, 10, f"Logger.debug should be called at least 10 times.")

if __name__ == '__main__':
    unittest.main()
