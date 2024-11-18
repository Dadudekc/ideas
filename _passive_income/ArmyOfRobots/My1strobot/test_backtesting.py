import pytest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime, timezone
from tslabot import Strategy, Portfolio, Backtester

# Mock configuration for the strategy
@pytest.fixture
def strategy_config():
    return {
        "vwap_session": "RTH",
        "ema_length": 8,
        "rsi_length": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "adx_length": 14,
        "volume_threshold_length": 20,
        "atr_length": 14,
        "risk_percent": 0.5,
        "profit_target_percent": 15.0,
        "stop_multiplier": 2.0,
        "trail_multiplier": 1.5,
        "timeframe": "1D",
        "limit": 1000
    }

# Mock historical data
@pytest.fixture
def mock_data():
    data = {
        "timestamp": [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
            datetime(2024, 1, 4, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
        ],
        "open": [100, 102, 101, 103, 102],
        "high": [105, 107, 106, 108, 107],
        "low": [99, 101, 100, 102, 101],
        "close": [104, 106, 105, 107, 106],
        "volume": [1000, 1100, 1050, 1200, 1150]
    }
    return pd.DataFrame(data)

# Test for backtesting workflow
def test_backtest_workflow(strategy_config, mock_data):
    # Initialize Strategy
    strategy = Strategy(
        vwap_session=strategy_config["vwap_session"],
        ema_length=strategy_config["ema_length"],
        rsi_length=strategy_config["rsi_length"],
        rsi_overbought=strategy_config["rsi_overbought"],
        rsi_oversold=strategy_config["rsi_oversold"],
        macd_fast=strategy_config["macd_fast"],
        macd_slow=strategy_config["macd_slow"],
        macd_signal=strategy_config["macd_signal"],
        adx_length=strategy_config["adx_length"],
        volume_threshold_length=strategy_config["volume_threshold_length"],
        atr_length=strategy_config["atr_length"],
        risk_percent=strategy_config["risk_percent"],
        profit_target_percent=strategy_config["profit_target_percent"],
        stop_multiplier=strategy_config["stop_multiplier"],
        trail_multiplier=strategy_config["trail_multiplier"],
        timeframe=strategy_config["timeframe"],
        limit=strategy_config["limit"]
    )

    # Mock Portfolio
    portfolio = Portfolio(initial_balance=100000)

    # Mock Logger
    logger = MagicMock()

    # Backtester instance
    backtester = Backtester(
        api=None,  # Not needed for this test
        symbol="TSLA",
        timeframe="1D",
        limit=1000,
        strategy=strategy,
        logger=logger,
        portfolio=portfolio,
        log_callback=MagicMock()
    )

    # Mock the load_data method
    backtester.load_data = MagicMock(return_value=mock_data)

    # Run backtesting
    backtester.run()

    # Assertions
    assert portfolio.balance == 100000, "Portfolio balance should remain unchanged for mock data without signals."
    assert len(portfolio.trade_history) == 0, "No trades should have been executed with the given mock data."
    logger.info.assert_any_call("Backtesting complete. Summary of results:")
    logger.info.assert_any_call("Final Portfolio Balance: $100000.00")
    logger.info.assert_any_call("Total Trades Executed: 0")

# Test indicator calculations
def test_indicator_calculations(strategy_config, mock_data):
    strategy = Strategy(
        vwap_session=strategy_config["vwap_session"],
        ema_length=strategy_config["ema_length"],
        rsi_length=strategy_config["rsi_length"],
        rsi_overbought=strategy_config["rsi_overbought"],
        rsi_oversold=strategy_config["rsi_oversold"],
        macd_fast=strategy_config["macd_fast"],
        macd_slow=strategy_config["macd_slow"],
        macd_signal=strategy_config["macd_signal"],
        adx_length=strategy_config["adx_length"],
        volume_threshold_length=strategy_config["volume_threshold_length"],
        atr_length=strategy_config["atr_length"],
        risk_percent=strategy_config["risk_percent"],
        profit_target_percent=strategy_config["profit_target_percent"],
        stop_multiplier=strategy_config["stop_multiplier"],
        trail_multiplier=strategy_config["trail_multiplier"],
        timeframe=strategy_config["timeframe"],
        limit=strategy_config["limit"]
    )

    # Calculate indicators
    df_with_indicators = strategy._calculate_indicators(mock_data)

    # Check if indicators are correctly added
    assert "vwap" in df_with_indicators.columns, "VWAP column should be calculated."
    assert "ema" in df_with_indicators.columns, "EMA column should be calculated."
    assert "rsi" in df_with_indicators.columns, "RSI column should be calculated."
    assert "macd_line" in df_with_indicators.columns, "MACD line should be calculated."
    assert "signal_line" in df_with_indicators.columns, "Signal line should be calculated."
    assert "adx" in df_with_indicators.columns, "ADX column should be calculated."
    assert "atr" in df_with_indicators.columns, "ATR column should be calculated."

    # Ensure no NaN values in the calculated indicators
    assert not df_with_indicators["vwap"].isna().all(), "VWAP values should not be all NaN."
    assert not df_with_indicators["ema"].isna().all(), "EMA values should not be all NaN."
    assert not df_with_indicators["rsi"].isna().all(), "RSI values should not be all NaN."
