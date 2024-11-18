import sys
import os
import pandas as pd
import time
import logging
import yaml
from datetime import datetime, timezone, timedelta
import alpaca_trade_api as tradeapi
import ta
import yfinance as yf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QComboBox,
    QMessageBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ----------------------------- Utility Functions ----------------------------- #

def load_config(config_file):
    """
    Load configuration from a YAML file. If the file does not exist, a default configuration
    is created, saved, and returned.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data loaded from the file.
    """
    # Default configuration
    default_config = {
        "mode": "backtest",
        "symbol": "TSLA",
        "timeframe": "1D",
        "limit": 1000,
        "api_key": "",
        "api_secret": "",
        "base_url": "https://paper-api.alpaca.markets",
        "strategy": {
            "vwap_session": "RTH",  # String parameter
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
            "trail_multiplier": 1.5
        }
    }

    # Check if configuration file exists
    if not os.path.exists(config_file):
        try:
            # Create and save the default configuration
            with open(config_file, "w") as file:
                yaml.dump(default_config, file, default_flow_style=False)
            logging.info(f"Default configuration created at {config_file}.")
        except Exception as e:
            logging.error(f"Error creating default configuration: {e}")
            raise

    # Load the configuration from the file
    try:
        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)
            # Validate configuration
            if not isinstance(config_data, dict):
                raise ValueError("Configuration file is malformed or empty.")
            return config_data
    except Exception as e:
        logging.error(f"Error loading configuration file {config_file}: {e}")
        raise

def save_config(config_file, config_data):
    """
    Save configuration data to a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.
        config_data (dict): Configuration data to be saved.

    Returns:
        None
    """
    try:
        with open(config_file, "w") as file:
            yaml.dump(config_data, file, default_flow_style=False)
        logging.info(f"Configuration saved successfully to {config_file}.")
    except Exception as e:
        logging.error(f"Error saving configuration to {config_file}: {e}")
        raise

def setup_logger():
    """
    Set up a logger that logs to both console and GUI.
    """
    logger = logging.getLogger("TSLA_Trading_Bot")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
    logger.handlers = []  # Reset handlers

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# ----------------------------- Portfolio Class ----------------------------- #

class Portfolio:
    """
    Simulates a trading portfolio by tracking balance, positions, and P&L.
    """
    def __init__(self, initial_balance=100000):
        self.balance = initial_balance
        self.positions = {}  # key: symbol, value: {'quantity': int, 'entry_price': float}
        self.trade_history = []  # List of executed trades

    def buy(self, symbol, price, quantity):
        total_cost = price * quantity
        if self.balance >= total_cost:
            self.balance -= total_cost
            if symbol in self.positions:
                total_quantity = self.positions[symbol]['quantity'] + quantity
                avg_price = ((self.positions[symbol]['entry_price'] * self.positions[symbol]['quantity']) + (price * quantity)) / total_quantity
                self.positions[symbol]['quantity'] = total_quantity
                self.positions[symbol]['entry_price'] = avg_price
            else:
                self.positions[symbol] = {'quantity': quantity, 'entry_price': price}
            self.trade_history.append({
                'type': 'BUY',
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'time': datetime.now(timezone.utc)
            })
            return True
        else:
            return False

    def sell(self, symbol, price, quantity):
        if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
            total_revenue = price * quantity
            self.balance += total_revenue
            self.positions[symbol]['quantity'] -= quantity
            if self.positions[symbol]['quantity'] == 0:
                del self.positions[symbol]
            self.trade_history.append({
                'type': 'SELL',
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'time': datetime.now(timezone.utc)
            })
            return True
        else:
            return False

    def get_position(self, symbol):
        return self.positions.get(symbol, {'quantity': 0, 'entry_price': 0.0})

    def get_total_equity(self, current_prices):
        equity = self.balance
        for symbol, pos in self.positions.items():
            equity += pos['quantity'] * current_prices.get(symbol, 0.0)
        return equity

# ----------------------------- Strategy Class ----------------------------- #

class Strategy:
    def __init__(self, **kwargs):
        # Initialize strategy parameters with validation
        self.vwap_session = kwargs.get("vwap_session", "RTH")  # String parameter
        self.ema_length = self._validate_param(kwargs.get("ema_length", 8), "ema_length")
        self.rsi_length = self._validate_param(kwargs.get("rsi_length", 14), "rsi_length")
        self.rsi_overbought = self._validate_param(kwargs.get("rsi_overbought", 70), "rsi_overbought")
        self.rsi_oversold = self._validate_param(kwargs.get("rsi_oversold", 30), "rsi_oversold")
        self.macd_fast = self._validate_param(kwargs.get("macd_fast", 12), "macd_fast")
        self.macd_slow = self._validate_param(kwargs.get("macd_slow", 26), "macd_slow")
        self.macd_signal = self._validate_param(kwargs.get("macd_signal", 9), "macd_signal")
        self.adx_length = self._validate_param(kwargs.get("adx_length", 14), "adx_length")
        self.volume_threshold_length = self._validate_param(kwargs.get("volume_threshold_length", 20), "volume_threshold_length")
        self.atr_length = self._validate_param(kwargs.get("atr_length", 14), "atr_length")
        self.risk_percent = kwargs.get("risk_percent", 1.0)
        self.profit_target_percent = kwargs.get("profit_target_percent", 2.0)
        self.stop_multiplier = kwargs.get("stop_multiplier", 1.5)
        self.trail_multiplier = kwargs.get("trail_multiplier", 1.0)
        self.timeframe = kwargs.get("timeframe", "1D")
        self.limit = kwargs.get("limit", 1000)
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def _validate_param(self, value, name, expected_type=int):
        """
        Validate that the parameter is of the expected type and within valid range.
        """
        if not isinstance(value, expected_type):
            self.logger.error(f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}")
            raise TypeError(f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}")
        if expected_type == int and value <= 0:
            self.logger.error(f"Parameter '{name}' must be a positive integer, got {value}")
            raise ValueError(f"Parameter '{name}' must be a positive integer, got {value}")
        if expected_type == float and value <= 0.0:
            self.logger.error(f"Parameter '{name}' must be a positive float, got {value}")
            raise ValueError(f"Parameter '{name}' must be a positive float, got {value}")
        return value

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators and add them as columns to the DataFrame.
        """
        df = df.copy()  # Avoid SettingWithCopyWarning
        self.logger.debug("Starting indicator calculations.")

        try:
            # Validate input data
            required_columns = ['high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Input DataFrame is missing required columns: {missing_columns}")
                raise ValueError(f"Missing columns: {missing_columns}")

            # Determine the maximum window size required among all indicators
            window_sizes: List[int] = [
                self.ema_length,
                self.rsi_length,
                self.macd_slow,  # MACD slow window is typically the largest
                self.adx_length,
                self.volume_threshold_length,
                self.atr_length
            ]
            max_window = max(window_sizes)
            self.logger.debug(f"Maximum window size required: {max_window}")

            # Check if the DataFrame has enough rows
            if len(df) < max_window + 1:
                self.logger.error(
                    f"Not enough data to calculate indicators. "
                    f"Required: {max_window + 1}, Provided: {len(df)}"
                )
                raise ValueError(
                    f"Not enough data to calculate indicators. "
                    f"Required: {max_window + 1}, Provided: {len(df)}"
                )

            # Calculate Indicators
            self._calculate_vwap(df)
            self._calculate_ema(df)
            self._calculate_rsi(df)
            self._calculate_macd(df)
            self._calculate_adx(df)
            self._calculate_volume_threshold(df)
            self._calculate_atr(df)

            # Log the DataFrame after adding indicators
            self.logger.debug(f"Indicators calculated successfully. DataFrame tail:\n{df.tail()}")

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            raise

        return df

    def _calculate_vwap(self, df: pd.DataFrame):
        """Calculate Volume Weighted Average Price (VWAP)."""
        self.logger.debug("Calculating VWAP.")
        try:
            vwap_indicator = ta.volume.VolumeWeightedAveragePrice(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14  # VWAP typically uses a fixed window
            )
            df['vwap'] = vwap_indicator.volume_weighted_average_price()
            self.logger.debug("VWAP calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            raise

    def _calculate_ema(self, df: pd.DataFrame):
        """Calculate Exponential Moving Average (EMA)."""
        self.logger.debug("Calculating EMA.")
        try:
            ema_indicator = ta.trend.EMAIndicator(close=df['close'], window=self.ema_length)
            df['ema'] = ema_indicator.ema_indicator()
            self.logger.debug("EMA calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            raise

    def _calculate_rsi(self, df: pd.DataFrame):
        """Calculate Relative Strength Index (RSI)."""
        self.logger.debug("Calculating RSI.")
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_length)
            df['rsi'] = rsi_indicator.rsi()
            self.logger.debug("RSI calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            raise

    def _calculate_macd(self, df: pd.DataFrame):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        self.logger.debug("Calculating MACD.")
        try:
            macd = ta.trend.MACD(
                close=df['close'],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal
            )
            df['macd_line'] = macd.macd()
            df['signal_line'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            self.logger.debug("MACD calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            raise

    def _calculate_adx(self, df: pd.DataFrame):
        """Calculate Average Directional Index (ADX)."""
        self.logger.debug("Calculating ADX.")
        try:
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'], low=df['low'], close=df['close'], window=self.adx_length
            )
            df['adx'] = adx_indicator.adx()
            self.logger.debug("ADX calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            raise

    def _calculate_volume_threshold(self, df: pd.DataFrame):
        """Calculate Volume Threshold based on rolling mean."""
        self.logger.debug("Calculating Volume Threshold.")
        try:
            df['volume_threshold'] = df['volume'].rolling(window=self.volume_threshold_length).mean()
            self.logger.debug("Volume Threshold calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating Volume Threshold: {e}")
            raise

    def _calculate_atr(self, df: pd.DataFrame):
        """Calculate Average True Range (ATR)."""
        self.logger.debug("Calculating ATR.")
        try:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=self.atr_length
            )
            df['atr'] = atr_indicator.average_true_range()
            self.logger.debug(f"ATR calculated successfully. Sample values:\n{df['atr'].tail()}")

            # Validate ATR
            if df['atr'].isnull().all():
                self.logger.error("ATR column contains only NaN values.")
                raise ValueError("ATR calculation failed or contains invalid data.")
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            raise

    def evaluate(self, df: pd.DataFrame) -> list:
        """
        Evaluate the latest indicators and generate trading signals.

        Parameters:
            df (pd.DataFrame): The DataFrame slice up to the current point in time.

        Returns:
            list: A list containing trading signals, e.g., ['BUY'], ['SELL'], or [].
        """
        self.logger.debug("Evaluating strategy based on indicators.")
        signals = []

        if df.empty:
            self.logger.debug("DataFrame slice is empty. No action taken.")
            return signals

        # Ensure indicators are calculated
        required_indicators = ['rsi', 'adx']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        if missing_indicators:
            self.logger.error(f"Missing indicators for evaluation: {missing_indicators}")
            return signals

        # Get the latest indicator values
        latest = df.iloc[-1]

        try:
            # Example Strategy Logic:
            # Buy signal: RSI below oversold and ADX above a threshold (e.g., 25)
            if (latest['rsi'] < self.rsi_oversold) and (latest['adx'] > 25):
                signals.append('BUY')
                self.logger.debug("Buy signal generated.")

            # Sell signal: RSI above overbought and ADX above a threshold (e.g., 25)
            if (latest['rsi'] > self.rsi_overbought) and (latest['adx'] > 25):
                signals.append('SELL')
                self.logger.debug("Sell signal generated.")

        except KeyError as e:
            self.logger.error(f"Missing indicator for evaluation: {e}")

        return signals

# ----------------------------- Backtester Class ----------------------------- #

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
        try:
            data_with_indicators = self.strategy._calculate_indicators(data)
        except Exception as e:
            self.logger.error(f"Indicator calculation failed during backtest: {e}")
            self.log_callback.emit(f"Indicator calculation failed during backtest: {e}")
            return

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

# ----------------------------- PaperTrader Class ----------------------------- #

class PaperTrader(QThread):
    """
    Paper trading class to simulate live trading without real money.
    """
    log_signal = pyqtSignal(str)

    def __init__(self, api, symbol, strategy, logger, portfolio):
        super().__init__()
        self.api = api
        self.symbol = symbol
        self.strategy = strategy
        self.logger = logger
        self.portfolio = portfolio
        self.running = True
        self.historical_data = self.load_initial_data()

    def load_initial_data(self):
        """
        Load initial historical data to calculate indicators.
        """
        self.log_signal.emit(f"Fetching initial historical data for {self.symbol}...")
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=365)  # Fetch last 1 year of data

            # Corrected datetime formatting without fractional seconds and with 'Z'
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')

            barset = self.api.get_bars(
                self.symbol,
                self.strategy.timeframe,
                start=start_str,
                end=end_str,
                limit=self.strategy.limit
            )
            data = []
            for bar in barset:
                data.append({
                    "timestamp": bar.t,
                    "open": bar.o,
                    "high": bar.h,
                    "low": bar.l,
                    "close": bar.c,
                    "volume": bar.v
                })
            df = pd.DataFrame(data)
            self.log_signal.emit("Initial historical data fetched successfully from Alpaca.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching initial historical data: {e}")
            self.log_signal.emit(f"Error fetching initial historical data: {e}")
            # Attempt to fetch data using yfinance as a backup
            self.logger.info("Attempting to fetch data using yfinance...")
            self.log_signal.emit("Attempting to fetch data using yfinance...")
            try:
                yf_data = yf.download(self.symbol, start=start_str, end=end_str, interval='1d')
                if yf_data.empty:
                    self.logger.error("YFinance returned no data.")
                    self.log_signal.emit("YFinance returned no data.")
                    return pd.DataFrame()
                yf_data.reset_index(inplace=True)
                yf_data.rename(columns={'Date': 'timestamp'}, inplace=True)
                df_backup = yf_data[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(
                    columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
                )
                self.logger.info("Historical data fetched successfully using yfinance.")
                self.log_signal.emit("Historical data fetched successfully using yfinance.")
                return df_backup
            except Exception as yf_e:
                self.logger.error(f"Error fetching data using yfinance: {yf_e}")
                self.log_signal.emit(f"Error fetching data using yfinance: {yf_e}")
                return pd.DataFrame()

    def fetch_live_data(self):
        """
        Fetch live market data from Alpaca API.
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
            return data
        except Exception as e:
            self.logger.error(f"Error fetching live data: {e}")
            self.log_signal.emit(f"Error fetching live data: {e}")
            return None

    def run(self):
        """
        Execute the paper trading process.
        """
        self.log_signal.emit("Paper trading started...")
        try:
            while self.running:
                live_data = self.fetch_live_data()
                if live_data is not None:
                    # Convert live data to a DataFrame for consistency
                    live_data_df = pd.DataFrame([live_data])

                    # Append new data
                    self.historical_data = pd.concat([self.historical_data, live_data_df], ignore_index=True)

                    # Keep only the necessary window size for indicators
                    window_sizes = [
                        self.strategy.ema_length,
                        self.strategy.rsi_length,
                        self.strategy.macd_slow,
                        self.strategy.adx_length,
                        self.strategy.volume_threshold_length,
                        self.strategy.atr_length
                    ]
                    max_window = max(window_sizes)
                    required_length = max_window + 1
                    self.historical_data = self.historical_data.tail(required_length)

                    # Ensure there is enough data to evaluate
                    if len(self.historical_data) < required_length:
                        self.log_signal.emit("Insufficient data for indicator calculations. Waiting for more data...")
                        time.sleep(60)
                        continue

                    # Calculate indicators for the updated data
                    try:
                        data_with_indicators = self.strategy._calculate_indicators(self.historical_data)
                    except Exception as e:
                        self.logger.error(f"Indicator calculation failed during paper trade: {e}")
                        self.log_signal.emit(f"Indicator calculation failed during paper trade: {e}")
                        time.sleep(60)
                        continue

                    # Evaluate signals
                    signals = self.strategy.evaluate(data_with_indicators)
                    for signal in signals:
                        self.execute_trade(signal, live_data)

                time.sleep(60)  # Pause for 1 minute
        except Exception as e:
            self.log_signal.emit(f"Error in paper trading loop: {e}")
            self.logger.error(f"Error in paper trading loop: {e}")
        finally:
            self.log_signal.emit("Paper trading stopped.")
            self.logger.info("Paper trading stopped.")

    def execute_trade(self, decision, data):
        """
        Simulate trade execution based on the decision.
        """
        price = data['close']
        atr = self.calculate_atr()
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
                self.log_signal.emit("Risk per share is zero. Skipping BUY trade.")
                return
            quantity = int(risk / risk_per_share)
            if quantity <= 0:
                self.logger.warning("Calculated quantity is zero or negative. Skipping trade.")
                self.log_signal.emit("Calculated quantity is zero or negative. Skipping BUY trade.")
                return

            # Execute BUY
            success = self.portfolio.buy(self.symbol, price, quantity)
            if success:
                self.logger.info(f"BUY {quantity} shares at ${price:.2f}")
                self.log_signal.emit(f"BUY {quantity} shares at ${price:.2f}")
                self.portfolio.trade_history[-1]['stop_loss'] = stop_loss
                self.portfolio.trade_history[-1]['take_profit'] = take_profit
            else:
                self.logger.warning("Insufficient balance to execute BUY.")
                self.log_signal.emit("Insufficient balance to execute BUY.")

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
                    self.log_signal.emit(f"SELL {quantity} shares at ${price:.2f}")
                    self.portfolio.trade_history[-1]['stop_loss'] = stop_loss
                    self.portfolio.trade_history[-1]['take_profit'] = take_profit
                else:
                    self.logger.warning("No position to SELL.")
                    self.log_signal.emit("No position to SELL.")
            else:
                self.logger.warning("No position to SELL.")
                self.log_signal.emit("No position to SELL.")

    def calculate_atr(self):
        """
        Calculate the current ATR value using the historical data.
        """
        if len(self.historical_data) < self.strategy.atr_length:
            return 0.0
        atr_indicator = ta.volatility.AverageTrueRange(
            high=self.historical_data['high'],
            low=self.historical_data['low'],
            close=self.historical_data['close'],
            window=self.strategy.atr_length
        )
        atr = atr_indicator.average_true_range().iloc[-1]
        return atr if not pd.isna(atr) else 0.0

    def stop_trading(self):
        """
        Stop the trading thread.
        """
        self.running = False
        self.quit()
        self.wait()

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

# ----------------------------- Worker Threads ----------------------------- #

class BacktestThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, backtester):
        super().__init__()
        self.backtester = backtester
        self.backtester.log_callback = self.log_signal

    def run(self):
        self.backtester.run()

class PaperTradeThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, trader):
        super().__init__()
        self.trader = trader
        self.trader.log_signal = self.log_signal

    def run(self):
        self.trader.run()

# ----------------------------- TradingBotGUI Class ----------------------------- #

class TradingBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tesla Trading Bot")
        self.setGeometry(100, 100, 800, 600)
        self.logger = setup_logger()

        # Initialize Portfolio
        self.portfolio = Portfolio(initial_balance=100000)

        # Load Configuration
        self.config_file = "config.yaml"
        self.config = load_config(self.config_file)

        # Initialize Threads
        self.backtest_thread = None
        self.paper_trade_thread = None

        # Setup UI
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # API Credentials
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setText(self.config.get("api_key", ""))
        api_layout.addWidget(self.api_key_input)

        api_layout.addWidget(QLabel("API Secret:"))
        self.api_secret_input = QLineEdit()
        self.api_secret_input.setText(self.config.get("api_secret", ""))
        self.api_secret_input.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.api_secret_input)

        api_layout.addWidget(QLabel("Base URL:"))
        self.base_url_input = QLineEdit()
        self.base_url_input.setText(self.config.get("base_url", "https://paper-api.alpaca.markets"))
        api_layout.addWidget(self.base_url_input)

        layout.addLayout(api_layout)

        # Save Credentials Button
        save_credentials_button = QPushButton("Save Credentials")
        save_credentials_button.clicked.connect(self.save_credentials)
        layout.addWidget(save_credentials_button)

        # Trading Mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["backtest", "paper_trade", "live_trade"])
        self.mode_combo.setCurrentText(self.config.get("mode", "backtest"))
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Strategy Parameters
        strategy_group = QWidget()
        strategy_layout = QVBoxLayout()

        # VWAP Session
        vwap_layout = QHBoxLayout()
        vwap_layout.addWidget(QLabel("VWAP Session:"))
        self.vwap_session_combo = QComboBox()
        self.vwap_session_combo.addItems(["RTH", "Full"])
        self.vwap_session_combo.setCurrentText(self.config.get("strategy", {}).get("vwap_session", "RTH"))
        vwap_layout.addWidget(self.vwap_session_combo)
        strategy_layout.addLayout(vwap_layout)

        # EMA Length
        ema_layout = QHBoxLayout()
        ema_layout.addWidget(QLabel("EMA Length:"))
        self.ema_length_spin = QSpinBox()
        self.ema_length_spin.setRange(1, 100)
        self.ema_length_spin.setValue(self.config.get("strategy", {}).get("ema_length", 8))
        ema_layout.addWidget(self.ema_length_spin)
        strategy_layout.addLayout(ema_layout)

        # RSI Parameters
        rsi_layout = QHBoxLayout()
        rsi_layout.addWidget(QLabel("RSI Length:"))
        self.rsi_length_spin = QSpinBox()
        self.rsi_length_spin.setRange(1, 100)
        self.rsi_length_spin.setValue(self.config.get("strategy", {}).get("rsi_length", 14))
        rsi_layout.addWidget(self.rsi_length_spin)

        rsi_layout.addWidget(QLabel("RSI Overbought:"))
        self.rsi_overbought_spin = QSpinBox()
        self.rsi_overbought_spin.setRange(1, 100)
        self.rsi_overbought_spin.setValue(self.config.get("strategy", {}).get("rsi_overbought", 70))
        rsi_layout.addWidget(self.rsi_overbought_spin)

        rsi_layout.addWidget(QLabel("RSI Oversold:"))
        self.rsi_oversold_spin = QSpinBox()
        self.rsi_oversold_spin.setRange(1, 100)
        self.rsi_oversold_spin.setValue(self.config.get("strategy", {}).get("rsi_oversold", 30))
        rsi_layout.addWidget(self.rsi_oversold_spin)

        strategy_layout.addLayout(rsi_layout)

        # MACD Parameters
        macd_layout = QHBoxLayout()
        macd_layout.addWidget(QLabel("MACD Fast Length:"))
        self.macd_fast_spin = QSpinBox()
        self.macd_fast_spin.setRange(1, 100)
        self.macd_fast_spin.setValue(self.config.get("strategy", {}).get("macd_fast", 12))
        macd_layout.addWidget(self.macd_fast_spin)

        macd_layout.addWidget(QLabel("MACD Slow Length:"))
        self.macd_slow_spin = QSpinBox()
        self.macd_slow_spin.setRange(1, 100)
        self.macd_slow_spin.setValue(self.config.get("strategy", {}).get("macd_slow", 26))
        macd_layout.addWidget(self.macd_slow_spin)

        macd_layout.addWidget(QLabel("MACD Signal Smoothing:"))
        self.macd_signal_spin = QSpinBox()
        self.macd_signal_spin.setRange(1, 100)
        self.macd_signal_spin.setValue(self.config.get("strategy", {}).get("macd_signal", 9))
        macd_layout.addWidget(self.macd_signal_spin)

        strategy_layout.addLayout(macd_layout)

        # ADX Parameters
        adx_layout = QHBoxLayout()
        adx_layout.addWidget(QLabel("ADX Length:"))
        self.adx_length_spin = QSpinBox()
        self.adx_length_spin.setRange(1, 100)
        self.adx_length_spin.setValue(self.config.get("strategy", {}).get("adx_length", 14))
        adx_layout.addWidget(self.adx_length_spin)
        strategy_layout.addLayout(adx_layout)

        # Volume Threshold Parameters
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume Threshold Length:"))
        self.volume_threshold_spin = QSpinBox()
        self.volume_threshold_spin.setRange(1, 100)
        self.volume_threshold_spin.setValue(self.config.get("strategy", {}).get("volume_threshold_length", 20))
        volume_layout.addWidget(self.volume_threshold_spin)
        strategy_layout.addLayout(volume_layout)

        # ATR Parameters
        atr_layout = QHBoxLayout()
        atr_layout.addWidget(QLabel("ATR Length:"))
        self.atr_length_spin = QSpinBox()
        self.atr_length_spin.setRange(1, 100)
        self.atr_length_spin.setValue(self.config.get("strategy", {}).get("atr_length", 14))
        atr_layout.addWidget(self.atr_length_spin)
        strategy_layout.addLayout(atr_layout)

        # Risk and Reward Management
        risk_layout = QHBoxLayout()
        risk_layout.addWidget(QLabel("Risk Percent per Trade:"))
        self.risk_percent_spin = QDoubleSpinBox()
        self.risk_percent_spin.setRange(0.1, 100.0)
        self.risk_percent_spin.setSingleStep(0.1)
        self.risk_percent_spin.setValue(self.config.get("strategy", {}).get("risk_percent", 0.5))
        risk_layout.addWidget(self.risk_percent_spin)

        risk_layout.addWidget(QLabel("Profit Target Percent:"))
        self.profit_target_spin = QDoubleSpinBox()
        self.profit_target_spin.setRange(1.0, 100.0)
        self.profit_target_spin.setSingleStep(0.5)
        self.profit_target_spin.setValue(self.config.get("strategy", {}).get("profit_target_percent", 15.0))
        risk_layout.addWidget(self.profit_target_spin)

        strategy_layout.addLayout(risk_layout)

        # ATR Multipliers
        atr_mult_layout = QHBoxLayout()
        atr_mult_layout.addWidget(QLabel("Stop Multiplier (ATR):"))
        self.stop_multiplier_spin = QDoubleSpinBox()
        self.stop_multiplier_spin.setRange(0.1, 10.0)
        self.stop_multiplier_spin.setSingleStep(0.1)
        self.stop_multiplier_spin.setValue(self.config.get("strategy", {}).get("stop_multiplier", 2.0))
        atr_mult_layout.addWidget(self.stop_multiplier_spin)

        atr_mult_layout.addWidget(QLabel("Trailing Multiplier (ATR):"))
        self.trail_multiplier_spin = QDoubleSpinBox()
        self.trail_multiplier_spin.setRange(0.1, 10.0)
        self.trail_multiplier_spin.setSingleStep(0.1)
        self.trail_multiplier_spin.setValue(self.config.get("strategy", {}).get("trail_multiplier", 1.5))
        atr_mult_layout.addWidget(self.trail_multiplier_spin)

        strategy_layout.addLayout(atr_mult_layout)

        strategy_group.setLayout(strategy_layout)
        layout.addWidget(QLabel("Strategy Parameters:"))
        layout.addWidget(strategy_group)

        # Start and Stop Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Trading")
        self.start_button.clicked.connect(self.start_trading)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Trading")
        self.stop_button.clicked.connect(self.stop_trading)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        # Logs
        layout.addWidget(QLabel("Logs:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Portfolio Status
        portfolio_layout = QHBoxLayout()
        self.balance_label = QLabel(f"Balance: ${self.portfolio.balance:.2f}")
        portfolio_layout.addWidget(self.balance_label)

        self.positions_label = QLabel(f"Positions: {self.portfolio.positions}")
        portfolio_layout.addWidget(self.positions_label)

        layout.addLayout(portfolio_layout)

        central_widget.setLayout(layout)

    def save_credentials(self):
        """
        Save API credentials to config.yaml
        """
        api_key = self.api_key_input.text().strip()
        api_secret = self.api_secret_input.text().strip()
        base_url = self.base_url_input.text().strip()

        if not api_key or not api_secret or not base_url:
            QMessageBox.warning(self, "Input Error", "Please provide all API credentials.")
            return

        self.config["api_key"] = api_key
        self.config["api_secret"] = api_secret
        self.config["base_url"] = base_url

        save_config(self.config_file, self.config)
        QMessageBox.information(self, "Success", "API credentials saved successfully.")
        self.log_text.append("[LOG] API credentials saved successfully.")

    def start_trading(self):
        mode = self.mode_combo.currentText()
        symbol = self.config.get("symbol", "TSLA")
        timeframe = self.config.get("timeframe", "1D")
        limit = self.config.get("limit", 1000)

        # Update configuration from GUI inputs
        config = {
            "mode": mode,
            "api_key": self.api_key_input.text().strip(),
            "api_secret": self.api_secret_input.text().strip(),
            "base_url": self.base_url_input.text().strip(),
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "strategy": {
                "vwap_session": self.vwap_session_combo.currentText(),
                "ema_length": self.ema_length_spin.value(),
                "rsi_length": self.rsi_length_spin.value(),
                "rsi_overbought": self.rsi_overbought_spin.value(),
                "rsi_oversold": self.rsi_oversold_spin.value(),
                "macd_fast": self.macd_fast_spin.value(),
                "macd_slow": self.macd_slow_spin.value(),
                "macd_signal": self.macd_signal_spin.value(),
                "adx_length": self.adx_length_spin.value(),
                "volume_threshold_length": self.volume_threshold_spin.value(),
                "atr_length": self.atr_length_spin.value(),
                "risk_percent": self.risk_percent_spin.value(),
                "profit_target_percent": self.profit_target_spin.value(),
                "stop_multiplier": self.stop_multiplier_spin.value(),
                "trail_multiplier": self.trail_multiplier_spin.value()
            }
        }

        # Validate API credentials
        if not config["api_key"] or not config["api_secret"] or not config["base_url"]:
            QMessageBox.warning(self, "Input Error", "Please provide valid Alpaca API credentials.")
            return

        # Save updated config
        save_config(self.config_file, config)
        self.config = config  # Update the in-memory config

        # Initialize Alpaca API
        try:
            api = tradeapi.REST(
                config["api_key"],
                config["api_secret"],
                config["base_url"],
                api_version='v2'
            )
            account = api.get_account()
            self.log_text.append("[LOG] Connected to Alpaca API successfully.")
        except Exception as e:
            QMessageBox.critical(self, "API Connection Error", f"Failed to connect to Alpaca API: {e}")
            self.log_text.append(f"Failed to connect to Alpaca API: {e}")
            return

        # Initialize Strategy
        try:
            strategy = Strategy(
                vwap_session=config["strategy"]["vwap_session"],
                ema_length=config["strategy"]["ema_length"],
                rsi_length=config["strategy"]["rsi_length"],
                rsi_overbought=config["strategy"]["rsi_overbought"],
                rsi_oversold=config["strategy"]["rsi_oversold"],
                macd_fast=config["strategy"]["macd_fast"],
                macd_slow=config["strategy"]["macd_slow"],
                macd_signal=config["strategy"]["macd_signal"],
                adx_length=config["strategy"]["adx_length"],
                volume_threshold_length=config["strategy"]["volume_threshold_length"],
                atr_length=config["strategy"]["atr_length"],
                risk_percent=config["strategy"]["risk_percent"],
                profit_target_percent=config["strategy"]["profit_target_percent"],
                stop_multiplier=config["strategy"]["stop_multiplier"],
                trail_multiplier=config["strategy"]["trail_multiplier"],
                timeframe=config["timeframe"],
                limit=config["limit"]
            )
            self.log_text.append("[LOG] Strategy initialized successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Strategy Initialization Error", f"Failed to initialize strategy: {e}")
            self.log_text.append(f"Failed to initialize strategy: {e}")
            return

        if mode == "backtest":
            self.log_text.append("[LOG] Starting Backtest Mode...")
            backtester = Backtester(api, symbol, timeframe, limit, strategy, self.logger, self.portfolio, self.log_text)
            self.backtest_thread = BacktestThread(backtester)
            self.backtest_thread.log_signal.connect(self.update_log)
            self.backtest_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        elif mode in ["paper_trade", "live_trade"]:
            self.log_text.append(f"[LOG] Starting {'Paper Trading' if mode == 'paper_trade' else 'Live Trading'} Mode...")
            trader = PaperTrader(api, symbol, strategy, self.logger, self.portfolio)
            self.paper_trade_thread = PaperTradeThread(trader)
            self.paper_trade_thread.log_signal.connect(self.update_log)
            self.paper_trade_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            QMessageBox.warning(self, "Mode Error", f"Unknown mode selected: {mode}")

    def stop_trading(self):
        mode = self.mode_combo.currentText()
        if mode in ["paper_trade", "live_trade"] and self.paper_trade_thread:
            self.paper_trade_thread.trader.stop_trading()
            self.paper_trade_thread.quit()
            self.paper_trade_thread.wait()
            self.log_text.append("[LOG] Trading session stopped.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        else:
            QMessageBox.warning(self, "Stop Error", "No active trading session to stop.")

    def update_log(self, message):
        self.log_text.append(message)
        # Update portfolio status
        self.balance_label.setText(f"Balance: ${self.portfolio.balance:.2f}")
        self.positions_label.setText(f"Positions: {self.portfolio.positions}")

# ----------------------------- Main Entry Point ----------------------------- #

def main():
    app = QApplication(sys.argv)
    gui = TradingBotGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
