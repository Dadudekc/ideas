# backtester.py

import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
import yfinance as yf
from strategy import Strategy
from portfolio import Portfolio
from PyQt5.QtCore import QObject

class Backtester:
    """
    Backtesting class to simulate trading strategies on historical data.
    """
    def __init__(self, api, symbol, timeframe, limit, strategy, logger, portfolio, log_callback):
        self.api = api
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.strategy = strategy
        self.logger = logger
        self.portfolio = portfolio
        self.log_callback = log_callback

    def load_data(self):
        """
        Load historical market data from Alpaca API or fallback to yfinance.
        """
        self.logger.info(f"Fetching historical data for {self.symbol}...")
        self.log_callback.emit(f"Fetching historical data for {self.symbol}...")

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365)  # Fetch last 1 year of data

        try:
            # Attempt to fetch data from Alpaca
            barset = self.api.get_bars(
                self.symbol,
                self.timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=self.limit
            )
            data = [
                {
                    "timestamp": bar.t,
                    "open": bar.o,
                    "high": bar.h,
                    "low": bar.l,
                    "close": bar.c,
                    "volume": bar.v,
                }
                for bar in barset
            ]
            df = pd.DataFrame(data)
            self.logger.info("Historical data fetched successfully from Alpaca.")
            self.log_callback.emit("Historical data fetched successfully from Alpaca.")
        except Exception as e:
            # Log Alpaca failure and attempt yfinance
            self.logger.error(f"Error fetching historical data from Alpaca: {e}")
            self.log_callback.emit(f"Error fetching historical data from Alpaca: {e}")
            self.logger.info("Attempting to fetch data using yfinance...")
            self.log_callback.emit("Attempting to fetch data using yfinance...")

            try:
                # Fetch data using yfinance
                yf_data = yf.download(
                    self.symbol,
                    start=start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                if yf_data.empty:
                    self.logger.error("YFinance returned no data.")
                    self.log_callback.emit("YFinance returned no data.")
                    return pd.DataFrame()

                # Process yfinance data
                yf_data.reset_index(inplace=True)
                yf_data.rename(
                    columns={
                        'Date': 'timestamp',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume',
                    },
                    inplace=True
                )
                df = yf_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                self.logger.info("Historical data fetched successfully using yfinance.")
                self.log_callback.emit("Historical data fetched successfully using yfinance.")
            except Exception as yf_e:
                # Log yfinance failure and return empty DataFrame
                self.logger.error(f"Error fetching data using yfinance: {yf_e}")
                self.log_callback.emit(f"Error fetching data using yfinance: {yf_e}")
                return pd.DataFrame()

        # Validate required columns
        missing_columns = [col for col in ['high', 'low', 'close', 'volume'] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")

        return df

    def run(self):
        """
        Execute the backtesting process.
        """
        data = self.load_data()
        if data.empty:
            self.log_callback.emit("No data to backtest.")
            return
        self.logger.info("Starting backtesting...")
        self.log_callback.emit("Starting backtesting...")

        # Calculate indicators for the entire dataset upfront to optimize performance
        data_with_indicators = self.strategy.calculate_indicators(data)

        for index in range(1, len(data_with_indicators)):
            df_slice = data_with_indicators.iloc[:index+1]
            current_signals = self.strategy.evaluate(df_slice)

            for signal in current_signals:
                if signal == 'BUY':
                    self.execute_trade('BUY', df_slice.iloc[-1])
                elif signal == 'SELL':
                    self.execute_trade('SELL', df_slice.iloc[-1])

        self.report_results()

    def execute_trade(self, decision, data):
        """
        Simulate trade execution based on the decision.
        """
        price = data['close']
        atr = data['atr']
        risk = self.portfolio.balance * (self.strategy.risk_percent / 100)
        stop_loss = 0
        take_profit = 0
        quantity = 0

        if decision == 'BUY':
            # Calculate stop-loss and take-profit levels
            stop_loss = price - (atr * self.strategy.stop_multiplier)
            take_profit = price + (price * self.strategy.profit_target_percent / 100)

            # Calculate position size
            risk_per_share = price - stop_loss
            if risk_per_share == 0:
                self.logger.warning("Risk per share is zero. Skipping trade.")
                self.log_callback.emit("Risk per share is zero. Skipping BUY trade.")
                return
            quantity = int(risk / risk_per_share)
            if quantity <= 0:
                self.logger.warning("Calculated quantity is zero or negative. Skipping trade.")
                self.log_callback.emit("Calculated quantity is zero or negative. Skipping BUY trade.")
                return

            # Execute BUY
            success = self.portfolio.buy(self.symbol, price, quantity)
            if success:
                self.logger.info(f"BUY {quantity} shares at ${price:.2f}")
                self.log_callback.emit(f"BUY {quantity} shares at ${price:.2f}")
                self.portfolio.trade_history[-1]['stop_loss'] = stop_loss
                self.portfolio.trade_history[-1]['take_profit'] = take_profit
            else:
                self.logger.warning("Insufficient balance to execute BUY.")
                self.log_callback.emit("Insufficient balance to execute BUY.")

        elif decision == 'SELL':
            # For simplicity, SELL all positions (short selling not implemented)
            position = self.portfolio.get_position(self.symbol)
            quantity = position['quantity']
            if quantity > 0:
                take_profit = price - (price * self.strategy.profit_target_percent / 100)
                stop_loss = price + (atr * self.strategy.stop_multiplier)

                # Execute SELL
                success = self.portfolio.sell(self.symbol, price, quantity)
                if success:
                    self.logger.info(f"SELL {quantity} shares at ${price:.2f}")
                    self.log_callback.emit(f"SELL {quantity} shares at ${price:.2f}")
                    self.portfolio.trade_history[-1]['stop_loss'] = stop_loss
                    self.portfolio.trade_history[-1]['take_profit'] = take_profit
                else:
                    self.logger.warning("No position to SELL.")
                    self.log_callback.emit("No position to SELL.")
            else:
                self.logger.warning("No position to SELL.")
                self.log_callback.emit("No position to SELL.")

    def report_results(self):
        """
        Output the results of the backtest.
        """
        self.logger.info("Backtesting complete. Summary of results:")
        self.log_callback.emit("Backtesting complete. Summary of results:")
        self.logger.info(f"Final Portfolio Balance: ${self.portfolio.balance:.2f}")
        self.log_callback.emit(f"Final Portfolio Balance: ${self.portfolio.balance:.2f}")
        self.logger.info(f"Final Positions: {self.portfolio.positions}")
        self.log_callback.emit(f"Final Positions: {self.portfolio.positions}")
        self.logger.info(f"Total Trades Executed: {len(self.portfolio.trade_history)}")
        self.log_callback.emit(f"Total Trades Executed: {len(self.portfolio.trade_history)}")
        buys = len([trade for trade in self.portfolio.trade_history if trade['type'] == 'BUY'])
        sells = len([trade for trade in self.portfolio.trade_history if trade['type'] == 'SELL'])
        self.logger.info(f"Total BUY Trades: {buys}")
        self.log_callback.emit(f"Total BUY Trades: {buys}")
        self.logger.info(f"Total SELL Trades: {sells}")
        self.log_callback.emit(f"Total SELL Trades: {sells}")
