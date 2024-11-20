# strategy.py

import pandas as pd
import ta
import logging
from typing import Any
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyConfig(BaseModel):
    ema_length: int = Field(default=8, ge=1, description="Length of EMA calculation.")
    rsi_length: int = Field(default=14, ge=1, description="Length of RSI calculation.")
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0, description="RSI overbought threshold.")
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0, description="RSI oversold threshold.")
    macd_fast_window: int = Field(default=12, ge=1, description="Fast window for MACD calculation.")
    macd_slow_window: int = Field(default=26, ge=1, description="Slow window for MACD calculation.")
    macd_signal_window: int = Field(default=9, ge=1, description="Signal window for MACD calculation.")
    adx_window: int = Field(default=14, ge=1, description="ADX window for ADX calculation.")
    vwap_window: int = Field(default=14, ge=1, description="VWAP window for VWAP calculation.")
    bollinger_window: int = Field(default=20, ge=1, description="Window for Bollinger Bands calculation.")
    atr_window: int = Field(default=14, ge=1, description="ATR window for ATR calculation.")
    risk_percent: float = Field(default=0.5, ge=0.1, le=100.0, description="Risk percent per trade.")
    profit_target: float = Field(default=15.0, ge=1.0, le=100.0, description="Profit target percent.")
    stop_loss_multiplier: float = Field(default=2.0, ge=1.0, description="Stop loss multiplier based on ATR.")


class TradeSignal:
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'


class Strategy:
    """
    Strategy class encapsulates the parameters and logic required for trading strategies.
    """
    def __init__(self, config: StrategyConfig, logger: logging.Logger = logger):
        """
        Initialize the Strategy class with provided parameters.
        """
        self.config = config
        self.logger = logger

    def validate_dataframe(self, df: pd.DataFrame):
        """
        Validate the input DataFrame for required columns and sufficient data.
        """
        required_columns = {'close', 'high', 'low', 'volume'}
        missing = required_columns - set(df.columns)
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}. DataFrame must contain: {required_columns}")

        min_required = max(
            self.config.ema_length,
            self.config.rsi_length,
            self.config.macd_slow_window,
            self.config.bollinger_window,
            self.config.adx_window,
            self.config.atr_window
        )
        if len(df) < min_required:
            self.logger.error("Insufficient data rows for calculating indicators.")
            raise ValueError("Insufficient data rows for calculating indicators.")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators required for the strategy.
        """
        self.logger.debug("Calculating indicators.")
        self.validate_dataframe(df)

        # Exponential Moving Average
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=self.config.ema_length).ema_indicator()

        # Relative Strength Index
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=self.config.rsi_length).rsi()

        # Moving Average Convergence Divergence
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=self.config.macd_slow_window,
            window_fast=self.config.macd_fast_window,
            window_sign=self.config.macd_signal_window
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Average Directional Index
        adx = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.config.adx_window
        )
        df['adx'] = adx.adx()

        # Volume Weighted Average Price
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # 20-period Simple Moving Average of Volume
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()

        # Average True Range
        atr = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.config.atr_window
        )
        df['atr'] = atr.average_true_range()

        self.logger.debug("Indicators calculated successfully.")
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicators and entry conditions.
        """
        self.logger.info("Generating trading signals.")

        # Initialize signal column
        df['signal'] = TradeSignal.HOLD

        # Long Condition
        long_condition = (
            (df['close'] > df['vwap']) &
            (df['close'] > df['ema']) &
            (df['rsi'] > self.config.rsi_oversold) &
            (df['rsi'] < self.config.rsi_overbought) &
            (df['macd_diff'] > 0) &
            (df['macd_diff'].shift(1) <= 0)  # Crossover condition
        )
        
        self.logger.info(f"Long Condition:\n{long_condition}")

        # Short Condition
        short_condition = (
            (df['close'] < df['vwap']) &
            (df['close'] < df['ema']) &
            (df['rsi'] < self.config.rsi_overbought) &
            (df['rsi'] > self.config.rsi_oversold) &
            (df['macd_diff'] < 0) &
            (df['macd_diff'].shift(1) >= 0) &  # Crossunder condition
            (df['adx'] > 20) &
            (df['volume'] > df['volume_sma_20'])
        )
        
        self.logger.info(f"Short Condition:\n{short_condition}")

        df.loc[long_condition, 'signal'] = TradeSignal.BUY
        df.loc[short_condition, 'signal'] = TradeSignal.SELL

        self.logger.info("Trading signals generated.")
        return df

    def backtest_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest the strategy with risk and reward management.
        """
        self.logger.info("Starting backtest with single-trade handling.")

        # Initialize tracking columns
        df['position'] = 0  # 1 for LONG, -1 for SHORT, 0 for FLAT
        df['entry_price'] = pd.NA
        df['exit_price'] = pd.NA
        df['strategy_returns'] = 0.0
        df['equity'] = 100000.0  # Starting equity

        equity = 100000.0
        position = 0
        entry_price = 0.0
        stop_loss_price = 0.0
        profit_target_price = 0.0

        for idx in df.index:
            signal = df.at[idx, 'signal']
            price = df.at[idx, 'close']
            atr = df.at[idx, 'atr']

            self.logger.debug(f"Index {idx}: Signal={signal}, Position={position}, Equity={equity}")

            # Exit logic
            if position != 0:
                exit_trade = False
                pnl = 0.0

                if position == 1:  # Currently in LONG
                    if (
                        signal == TradeSignal.SELL or
                        price <= stop_loss_price or
                        price >= profit_target_price
                    ):
                        pnl = price - entry_price
                        exit_trade = True
                        self.logger.debug(
                            f"Exiting LONG at index {idx}, Price={price}, PnL={pnl:.2f}"
                        )
                elif position == -1:  # Currently in SHORT
                    if (
                        signal == TradeSignal.BUY or
                        price >= stop_loss_price or
                        price <= profit_target_price
                    ):
                        pnl = entry_price - price
                        exit_trade = True
                        self.logger.debug(
                            f"Exiting SHORT at index {idx}, Price={price}, PnL={pnl:.2f}"
                        )

                if exit_trade:
                    equity += pnl
                    df.at[idx, 'exit_price'] = price
                    df.at[idx, 'strategy_returns'] = pnl
                    position = 0  # Reset position to flat
                    entry_price = 0.0
                    stop_loss_price = 0.0
                    profit_target_price = 0.0
                    df.at[idx, 'position'] = position
                    df.at[idx, 'equity'] = equity
                    continue  # Skip entry logic in this iteration

            # Entry logic (only when flat)
            if position == 0:
                if signal == TradeSignal.BUY:
                    position = 1
                    entry_price = price
                    stop_loss_price = price - atr * self.config.stop_loss_multiplier
                    profit_target_price = price + price * (self.config.profit_target / 100)
                    df.at[idx, 'entry_price'] = entry_price
                    df.at[idx, 'position'] = position
                    self.logger.debug(
                        f"Entered LONG at index {idx}, Price={price}, "
                        f"StopLoss={stop_loss_price}, ProfitTarget={profit_target_price}"
                    )
                elif signal == TradeSignal.SELL:
                    position = -1
                    entry_price = price
                    stop_loss_price = price + atr * self.config.stop_loss_multiplier
                    profit_target_price = price - price * (self.config.profit_target / 100)
                    df.at[idx, 'entry_price'] = entry_price
                    df.at[idx, 'position'] = position
                    self.logger.debug(
                        f"Entered SHORT at index {idx}, Price={price}, "
                        f"StopLoss={stop_loss_price}, ProfitTarget={profit_target_price}"
                    )

            # Update equity for the current row
            df.at[idx, 'equity'] = equity

        self.logger.info("Backtest completed with single-trade handling.")
        return df
