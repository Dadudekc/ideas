# paper_trader.py

import pandas as pd
import ta
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime, timezone, timedelta
import yfinance as yf
import time
from strategy import Strategy
from portfolio import Portfolio
from typing import List, Optional


class PaperTrader(QThread):
    """
    Paper trading class to simulate live trading without real money.
    Inherits from QThread to run the trading loop in a separate thread.
    """
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str)

    def __init__(
        self,
        api,
        symbol: str,
        strategy: Strategy,
        logger: logging.Logger,
        portfolio: Portfolio,
        sleep_interval: int = 60,
        data_fetch_days: int = 365
    ):
        """
        Initialize the PaperTrader thread.

        :param api: API client for fetching live and historical data.
        :param symbol: Stock symbol to trade.
        :param strategy: Trading strategy instance.
        :param logger: Logger instance for logging events.
        :param portfolio: Portfolio instance to manage trades.
        :param sleep_interval: Time to wait between each trading loop iteration in seconds.
        :param data_fetch_days: Number of past days to fetch for historical data.
        """
        super().__init__()
        self.api = api
        self.symbol = symbol
        self.strategy = strategy
        self.logger = logger
        self.portfolio = portfolio
        self.running = True
        self.sleep_interval = sleep_interval
        self.data_fetch_days = data_fetch_days
        self.historical_data = self.load_initial_data()

    def load_initial_data(self) -> pd.DataFrame:
        """
        Load initial historical data to calculate indicators.

        Validates that all required columns are present in the data.
        """
        self.log_signal.emit(f"Fetching initial historical data for {self.symbol}...")
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.data_fetch_days)
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        try:
            # Fetch data using yfinance as a fallback
            yf_data = yf.download(
                self.symbol,
                start=start_str,
                end=end_str,
                interval='1d',
                progress=False
            )
            if yf_data.empty:
                raise ValueError("YFinance returned no data.")
            yf_data.reset_index(inplace=True)

            # Rename columns to standardize
            df = yf_data.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Validate required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing columns in fetched data: {missing_columns}")
                self.log_signal.emit(f"Missing columns in fetched data: {missing_columns}")
                raise ValueError(f"Missing columns: {missing_columns}")

            self.logger.info("Historical data fetched successfully using yfinance.")
            self.log_signal.emit("Historical data fetched successfully using yfinance.")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            self.log_signal.emit(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def fetch_live_data(self) -> Optional[pd.Series]:
        """
        Fetch live market data from the primary API.

        :return: Series containing live data or None if no new data.
        """
        try:
            bar = self.api.get_latest_bar(self.symbol)
            if bar is None:
                self.log_signal.emit("No new bar received.")
                return None
            data = {
                "timestamp": bar.t,
                "open": bar.o,
                "high": bar.h,
                "low": bar.l,
                "close": bar.c,
                "volume": bar.v
            }
            return pd.Series(data)
        except Exception as e:
            self.logger.error(f"Error fetching live data: {e}")
            self.log_signal.emit(f"Error fetching live data: {e}")
            return None

    def run(self):
        """
        Execute the paper trading process.
        Continuously fetches live data, updates indicators, evaluates signals, and executes trades.
        """
        self.log_signal.emit("Paper trading started...")
        self.logger.info("Paper trading started.")

        try:
            while self.running:
                live_data = self.fetch_live_data()
                if live_data is not None:
                    # Append new data using pd.concat instead of append
                    self.historical_data = pd.concat(
                        [self.historical_data, live_data.to_frame().T],
                        ignore_index=True
                    )

                    # Determine the maximum window size required for indicators
                    window_sizes = [
                        self.strategy.ema_length,
                        self.strategy.rsi_length,
                        self.strategy.macd_slow,
                        self.strategy.adx_length,
                        self.strategy.volume_threshold_length,
                        self.strategy.atr_length
                    ]
                    max_window = max(window_sizes)
                    required_length = max_window + 1  # Additional data point for accurate calculation

                    # Keep only the necessary window size
                    if len(self.historical_data) > required_length:
                        self.historical_data = self.historical_data.tail(required_length)

                    # Check if there's enough data to calculate indicators
                    if len(self.historical_data) < required_length:
                        self.log_signal.emit("Insufficient data for indicator calculations. Waiting for more data...")
                        self.logger.info("Insufficient data for indicator calculations. Waiting...")
                        time.sleep(self.sleep_interval)
                        continue

                    # Calculate indicators
                    try:
                        data_with_indicators = self.strategy.calculate_indicators(self.historical_data)
                    except Exception as e:
                        self.logger.error(f"Indicator calculation failed: {e}")
                        self.log_signal.emit(f"Indicator calculation failed: {e}")
                        time.sleep(self.sleep_interval)
                        continue

                    # Evaluate signals
                    try:
                        signals = self.strategy.evaluate(data_with_indicators)
                        for signal in signals:
                            self.execute_trade(signal, live_data)
                    except Exception as e:
                        self.logger.error(f"Error during signal evaluation: {e}")
                        self.log_signal.emit(f"Error during signal evaluation: {e}")

                time.sleep(self.sleep_interval)
        except Exception as e:
            self.logger.error(f"Error in paper trading loop: {e}")
            self.log_signal.emit(f"Error in paper trading loop: {e}")
        finally:
            self.log_signal.emit("Paper trading stopped.")
            self.logger.info("Paper trading stopped.")
            self.report_results()

    def execute_trade(self, decision: str, data: pd.Series):
        """
        Simulate trade execution based on the decision.

        :param decision: 'BUY' or 'SELL' signal.
        :param data: Series containing the latest market data.
        """
        price = data['close']
        atr = self.calculate_atr()
        risk = self.portfolio.balance * (self.strategy.risk_percent / 100)
        stop_loss = take_profit = quantity = 0

        if decision == 'BUY':
            # Calculate stop-loss and take-profit levels
            stop_loss = price - (atr * self.strategy.stop_multiplier)
            take_profit = price + (price * self.strategy.profit_target_percent / 100)

            # Calculate position size
            risk_per_share = price - stop_loss

            if risk_per_share <= 0:
                self.logger.warning("Risk per share is zero or negative. Skipping BUY trade.")
                self.log_signal.emit("Risk per share is zero or negative. Skipping BUY trade.")
                return

            quantity = int(risk / risk_per_share)
            if quantity <= 0:
                self.logger.warning("Calculated quantity is zero or negative. Skipping BUY trade.")
                self.log_signal.emit("Calculated quantity is zero or negative. Skipping BUY trade.")
                return

            # Execute BUY
            success = self.portfolio.buy(self.symbol, price, quantity)
            if success:
                self.logger.info(f"BUY {quantity} shares at ${price:.2f}")
                self.log_signal.emit(f"BUY {quantity} shares at ${price:.2f}")
                # Update the last trade with stop-loss and take-profit
                if not self.portfolio.trade_history.empty:
                    self.portfolio.trade_history.loc[
                        self.portfolio.trade_history.index[-1],
                        ['stop_loss', 'take_profit']
                    ] = [stop_loss, take_profit]
            else:
                self.logger.warning("Insufficient balance to execute BUY.")
                self.log_signal.emit("Insufficient balance to execute BUY.")

        elif decision == 'SELL':
            # SELL all positions (short selling not implemented)
            position = self.portfolio.get_position(self.symbol)
            quantity = position['quantity'] if position else 0
            if quantity > 0:
                take_profit = price - (price * self.strategy.profit_target_percent / 100)
                stop_loss = price + (atr * self.strategy.stop_multiplier)

                # Execute SELL
                success = self.portfolio.sell(self.symbol, price, quantity)
                if success:
                    self.logger.info(f"SELL {quantity} shares at ${price:.2f}")
                    self.log_signal.emit(f"SELL {quantity} shares at ${price:.2f}")
                    # Update the last trade with stop-loss and take-profit
                    if not self.portfolio.trade_history.empty:
                        self.portfolio.trade_history.loc[
                            self.portfolio.trade_history.index[-1],
                            ['stop_loss', 'take_profit']
                        ] = [stop_loss, take_profit]
                else:
                    self.logger.warning("No position to SELL.")
                    self.log_signal.emit("No position to SELL.")
            else:
                self.logger.warning("No position to SELL.")
                self.log_signal.emit("No position to SELL.")
        else:
            self.logger.warning(f"Unknown trade decision: {decision}")
            self.log_signal.emit(f"Unknown trade decision: {decision}")

    def calculate_atr(self) -> float:
        """
        Calculate the current ATR value using the historical data.

        :return: ATR value or 0.0 if not calculable.
        """
        try:
            if len(self.historical_data) < self.strategy.atr_length:
                self.logger.warning("Not enough data to calculate ATR.")
                return 0.0
            atr_indicator = ta.volatility.AverageTrueRange(
                high=self.historical_data['high'],
                low=self.historical_data['low'],
                close=self.historical_data['close'],
                window=self.strategy.atr_length
            )
            atr = atr_indicator.average_true_range().iloc[-1]
            return atr if pd.notna(atr) else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0

    def stop_trading(self):
        """
        Stop the trading thread gracefully.
        """
        self.running = False
        self.quit()
        self.wait()

    def report_results(self):
        """
        Output the results of the paper trading session.
        """
        if self.portfolio.trade_history.empty:
            summary = [
                "Backtesting complete. No trades were executed.",
                f"Final Portfolio Balance: ${self.portfolio.balance:.2f}",
                f"Final Positions: {self.portfolio.positions}"
            ]
        else:
            summary = [
                "Backtesting complete. Summary of results:",
                f"Final Portfolio Balance: ${self.portfolio.balance:.2f}",
                f"Final Positions: {self.portfolio.positions}",
                f"Total Trades Executed: {len(self.portfolio.trade_history)}",
                f"Total BUY Trades: {len(self.portfolio.trade_history[self.portfolio.trade_history['type'] == 'BUY'])}",
                f"Total SELL Trades: {len(self.portfolio.trade_history[self.portfolio.trade_history['type'] == 'SELL'])}"
            ]

        for line in summary:
            self.logger.info(line)
            self.result_signal.emit(line)
