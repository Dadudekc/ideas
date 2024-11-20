# test_strategy.py

import unittest
import pandas as pd
import numpy as np
from basicbot.strategy import Strategy, StrategyConfig, TradeSignal
import logging

class TestStrategy(unittest.TestCase):
    def setUp(self):
        # Configure logger for testing
        self.logger = logging.getLogger("TestStrategy")
        self.logger.setLevel(logging.CRITICAL)  # Suppress logs during testing

        # Default configuration
        self.config = StrategyConfig()
        self.strategy = Strategy(config=self.config, logger=self.logger)

        # Sample data for testing
        np.random.seed(0)
        dates = pd.date_range(start='2021-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'close': np.random.lognormal(mean=0, sigma=0.01, size=100).cumprod(),
            'high': np.random.lognormal(mean=0, sigma=0.01, size=100).cumprod(),
            'low': np.random.lognormal(mean=0, sigma=0.01, size=100).cumprod(),
            'volume': np.random.randint(100, 1000, size=100)
        }, index=dates)

    def test_validate_dataframe_missing_columns(self):
        # Remove 'volume' column to simulate missing data
        df_missing = self.df.drop(columns=['volume'])
        with self.assertRaises(ValueError) as context:
            self.strategy.validate_dataframe(df_missing)
        self.assertIn("Missing required columns", str(context.exception))

    def test_validate_dataframe_insufficient_data(self):
        # Create DataFrame with less rows than required
        df_insufficient = self.df.iloc[:5]
        with self.assertRaises(ValueError) as context:
            self.strategy.validate_dataframe(df_insufficient)
        self.assertIn("Insufficient data rows", str(context.exception))

    def test_calculate_indicators(self):
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        # Check if indicators are added
        expected_columns = [
            'ema', 'rsi', 'macd', 'macd_signal', 'macd_diff',
            'adx', 'vwap', 'volume_sma_20', 'atr'
        ]
        for col in expected_columns:
            self.assertIn(col, df_with_indicators.columns)
        # Check for no NaN in critical indicators after enough data
        self.assertFalse(df_with_indicators['ema'].isnull().all())
        self.assertFalse(df_with_indicators['rsi'].isnull().all())

    def test_generate_signals(self):
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        df_with_signals = self.strategy.generate_signals(df_with_indicators)
        # Check if 'signal' column is added
        self.assertIn('signal', df_with_signals.columns)
        # Check if signals are only among BUY, SELL, HOLD
        self.assertTrue(df_with_signals['signal'].isin(
            [TradeSignal.BUY, TradeSignal.SELL, TradeSignal.HOLD]
        ).all())

    def test_backtest_strategy_no_signals(self):
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        # Ensure no signals
        df_with_indicators['signal'] = TradeSignal.HOLD
        df_backtest = self.strategy.backtest_strategy(df_with_indicators)
        # Equity should remain unchanged
        self.assertTrue((df_backtest['equity'] == 100000.0).all())
        # No positions should be taken
        self.assertTrue((df_backtest['position'] == 0).all())

    def test_backtest_strategy_single_buy_sell(self):
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        df_with_signals = self.strategy.generate_signals(df_with_indicators)

        # Manually set a BUY signal at index 20 and SELL at index 30
        df_with_signals.at[self.df.index[20], 'signal'] = TradeSignal.BUY
        df_with_signals.at[self.df.index[30], 'signal'] = TradeSignal.SELL

        df_backtest = self.strategy.backtest_strategy(df_with_signals)

        # Check that a position was entered and exited
        self.assertEqual(df_backtest.at[self.df.index[20], 'position'], 1)
        self.assertEqual(df_backtest.at[self.df.index[30], 'position'], 0)
        self.assertEqual(df_backtest.at[self.df.index[30], 'exit_price'], df_with_signals.at[self.df.index[30], 'close'])
        # Equity should have been updated
        entry_price = df_with_signals.at[self.df.index[20], 'close']
        exit_price = df_with_signals.at[self.df.index[30], 'close']
        expected_pnl = exit_price - entry_price
        expected_equity = 100000.0 + expected_pnl
        self.assertEqual(df_backtest.at[self.df.index[30], 'equity'], expected_equity)

    def test_backtest_strategy_multiple_trades(self):
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        df_with_signals = self.strategy.generate_signals(df_with_indicators)

        # Set multiple BUY and SELL signals
        buy_indices = [10, 30, 50, 70]
        sell_indices = [15, 35, 55, 75]

        for buy in buy_indices:
            df_with_signals.at[self.df.index[buy], 'signal'] = TradeSignal.BUY
        for sell in sell_indices:
            df_with_signals.at[self.df.index[sell], 'signal'] = TradeSignal.SELL

        df_backtest = self.strategy.backtest_strategy(df_with_signals)

        # Verify that positions are entered and exited correctly
        for buy, sell in zip(buy_indices, sell_indices):
            self.assertEqual(df_backtest.at[self.df.index[buy], 'position'], 1)
            self.assertEqual(df_backtest.at[self.df.index[sell], 'position'], 0)
            self.assertEqual(df_backtest.at[self.df.index[sell], 'exit_price'], df_with_signals.at[self.df.index[sell], 'close'])

    def test_backtest_strategy_stop_loss(self):
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        df_with_signals = self.strategy.generate_signals(df_with_indicators)

        # Set a BUY signal and simulate price dropping below stop loss
        buy_index = 40
        df_with_signals.at[self.df.index[buy_index], 'signal'] = TradeSignal.BUY
        # Simulate a drop in price after buy_index to trigger stop loss
        drop_indices = range(buy_index + 1, buy_index + 5)
        for idx in drop_indices:
            df_with_signals.at[self.df.index[idx], 'close'] = df_with_signals.at[self.df.index[buy_index], 'close'] - \
                                                             2 * df_with_signals.at[self.df.index[buy_index], 'atr']

        df_backtest = self.strategy.backtest_strategy(df_with_signals)

        # Check that position was exited due to stop loss
        exit_idx = buy_index + 1
        self.assertEqual(df_backtest.at[self.df.index[buy_index], 'position'], 1)
        self.assertEqual(df_backtest.at[self.df.index[exit_idx], 'position'], 0)
        self.assertEqual(df_backtest.at[self.df.index[exit_idx], 'exit_price'], df_with_signals.at[self.df.index[exit_idx], 'close'])

    def test_backtest_strategy_profit_target(self):
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        df_with_signals = self.strategy.generate_signals(df_with_indicators)

        # Set a BUY signal and simulate price rising above profit target
        buy_index = 60
        df_with_signals.at[self.df.index[buy_index], 'signal'] = TradeSignal.BUY
        # Simulate a rise in price after buy_index to trigger profit target
        rise_indices = range(buy_index + 1, buy_index + 5)
        for idx in rise_indices:
            df_with_signals.at[self.df.index[idx], 'close'] = df_with_signals.at[self.df.index[buy_index], 'close'] + \
                                                             0.15 * df_with_signals.at[self.df.index[buy_index], 'close']

        df_backtest = self.strategy.backtest_strategy(df_with_signals)

        # Check that position was exited due to profit target
        exit_idx = buy_index + 1
        self.assertEqual(df_backtest.at[self.df.index[buy_index], 'position'], 1)
        self.assertEqual(df_backtest.at[self.df.index[exit_idx], 'position'], 0)
        self.assertEqual(df_backtest.at[self.df.index[exit_idx], 'exit_price'], df_with_signals.at[self.df.index[exit_idx], 'close'])

    def test_backtest_strategy_short_position(self):
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        df_with_signals = self.strategy.generate_signals(df_with_indicators)

        # Set a SELL signal at index 25 and BUY to close at index 35
        df_with_signals.at[self.df.index[25], 'signal'] = TradeSignal.SELL
        df_with_signals.at[self.df.index[35], 'signal'] = TradeSignal.BUY

        df_backtest = self.strategy.backtest_strategy(df_with_signals)

        # Check that a short position was entered and exited
        self.assertEqual(df_backtest.at[self.df.index[25], 'position'], -1)
        self.assertEqual(df_backtest.at[self.df.index[35], 'position'], 0)
        self.assertEqual(df_backtest.at[self.df.index[35], 'exit_price'], df_with_signals.at[self.df.index[35], 'close'])

    def test_full_strategy_flow(self):
        """
        Test the full flow: calculate indicators, generate signals, and backtest.
        """
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        df_with_signals = self.strategy.generate_signals(df_with_indicators)
        df_backtest = self.strategy.backtest_strategy(df_with_signals)

        # Basic checks
        self.assertIn('strategy_returns', df_backtest.columns)
        self.assertIn('equity', df_backtest.columns)
        self.assertEqual(len(df_backtest), len(self.df))
        # Equity should be >= starting equity if no losses
        # This is not always true, but we can check for no negative equity
        self.assertTrue((df_backtest['equity'] > 0).all())

    def test_invalid_config(self):
        # Test creating StrategyConfig with invalid parameters
        with self.assertRaises(ValueError):
            StrategyConfig(ema_length=0)  # ema_length must be >=1

        with self.assertRaises(ValueError):
            StrategyConfig(rsi_overbought=150)  # rsi_overbought must be <=100

        with self.assertRaises(ValueError):
            StrategyConfig(stop_loss_multiplier=0.5)  # stop_loss_multiplier must be >=1.0

    def test_no_indicator_overflow(self):
        # Ensure that indicator calculations do not produce infinities or NaNs beyond initial periods
        df_with_indicators = self.strategy.calculate_indicators(self.df.copy())
        self.assertFalse(np.isinf(df_with_indicators['ema']).any())
        self.assertFalse(np.isnan(df_with_indicators['ema']).all())

        self.assertFalse(np.isinf(df_with_indicators['rsi']).any())
        self.assertFalse(np.isnan(df_with_indicators['rsi']).all())

        self.assertFalse(np.isinf(df_with_indicators['macd']).any())
        self.assertFalse(np.isnan(df_with_indicators['macd']).all())

        self.assertFalse(np.isinf(df_with_indicators['adx']).any())
        self.assertFalse(np.isnan(df_with_indicators['adx']).all())

        self.assertFalse(np.isinf(df_with_indicators['vwap']).any())
        self.assertFalse(np.isnan(df_with_indicators['vwap']).all())

        self.assertFalse(np.isinf(df_with_indicators['volume_sma_20']).any())
        self.assertFalse(np.isnan(df_with_indicators['volume_sma_20']).all())

        self.assertFalse(np.isinf(df_with_indicators['atr']).any())
        self.assertFalse(np.isnan(df_with_indicators['atr']).all())

if __name__ == '__main__':
    unittest.main()
