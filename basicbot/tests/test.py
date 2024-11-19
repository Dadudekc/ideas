# trading_strategy_gui.py

import sys
import pandas as pd
import ta
import logging
import numpy as np
from typing import List
from pydantic import BaseModel, Field
import yfinance as yf
import pyqtgraph as pg
import threading
import time

from alpaca_trade_api import REST, StreamConn
from alpaca_trade_api.stream import Stream

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QDateEdit,
    QGridLayout, QSpinBox, QDoubleSpinBox, QTextEdit, QTableView, QProgressBar,
    QTabWidget, QGraphicsRectItem
)
from PyQt5.QtCore import Qt, QDate, QAbstractTableModel, QModelIndex, QTimer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Strategy Configuration using Pydantic
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

# Trade Signals
class TradeSignal:
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'

# Trade Table Model for QTableView
class TradeTableModel(QAbstractTableModel):
    def __init__(self, trades: List[dict]):
        super().__init__()
        self.trades = trades
        self.headers = ["Type", "Entry Date", "Entry Price", "Exit Date", "Exit Price", "PnL"]

    def rowCount(self, parent=QModelIndex()):
        return len(self.trades)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            trade = self.trades[index.row()]
            column = self.headers[index.column()]
            key = column.lower().replace(' ', '_')
            return str(trade.get(key, ''))
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            else:
                return section + 1
        return None

# Strategy Class
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

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators required for the strategy.
        """
        self.logger.debug("Calculating indicators.")

        required_columns = {'close', 'high', 'low', 'volume'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            self.logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        if len(df) < max(
            self.config.ema_length, self.config.rsi_length,
            self.config.macd_slow_window, self.config.bollinger_window,
            self.config.adx_window, self.config.atr_window
        ):
            self.logger.error("Insufficient data rows for calculating indicators.")
            raise ValueError("Insufficient data rows for calculating indicators.")

        # Exponential Moving Average
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=self.config.ema_length).ema_indicator()

        # Relative Strength Index
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=self.config.rsi_length).rsi()

        # Moving Average Convergence Divergence
        macd = ta.trend.MACD(close=df['close'],
                             window_fast=self.config.macd_fast_window,
                             window_slow=self.config.macd_slow_window,
                             window_sign=self.config.macd_signal_window)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Average Directional Index
        try:
            adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=self.config.adx_window)
            df['adx'] = adx.adx()
        except ValueError as e:
            self.logger.error(f"ADX calculation failed: {e}")
            df['adx'] = np.nan

        # Volume Weighted Average Price
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # 20-period Simple Moving Average of Volume
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()

        # Average True Range
        atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.config.atr_window)
        df['atr'] = atr.average_true_range()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=self.config.bollinger_window)
        df['bollinger_upper'] = bollinger.bollinger_hband()
        df['bollinger_lower'] = bollinger.bollinger_lband()

        self.logger.debug("Indicators calculated successfully.")
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicators and entry conditions.
        """
        self.logger.info("Generating trading signals.")

        # Initialize signal column
        df['signal'] = TradeSignal.HOLD

        # Long Condition with additional confirmation
        long_condition = (
            (df['close'] > df['vwap']) &
            (df['close'] > df['ema']) &
            (df['close'] < df['bollinger_upper']) &  # Close below upper Bollinger Band
            (df['rsi'] > 40) &
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))  # Crossover condition
        )

        # Short Condition with additional confirmation
        short_condition = (
            (df['close'] < df['vwap']) &
            (df['close'] < df['ema']) &
            (df['close'] > df['bollinger_lower']) &  # Close above lower Bollinger Band
            (df['rsi'] < 60) &
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1)) &  # Crossunder condition
            (df['adx'] > 20) &
            (df['volume'] > df['volume_sma_20'])
        )

        df.loc[long_condition, 'signal'] = TradeSignal.BUY
        df.loc[short_condition, 'signal'] = TradeSignal.SELL

        self.logger.info("Trading signals generated.")
        return df

    def backtest_strategy(self, df: pd.DataFrame, progress_callback=None) -> (pd.DataFrame, List[dict]):
        """
        Backtest the strategy with risk and reward management.
        """
        self.logger.info("Starting backtest.")

        # Initialize columns for tracking positions and equity
        df['position'] = 0  # 1 for LONG, -1 for SHORT, 0 for FLAT
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['strategy_returns'] = 0.0
        df['stop_loss'] = np.nan
        df['profit_target'] = np.nan
        df['equity'] = 100000.0  # Starting equity

        equity = 100000.0
        position = 0
        entry_price = 0.0
        stop_loss_price = 0.0
        profit_target_price = 0.0

        trades = []  # To store trade details

        for idx in range(len(df)):
            if progress_callback:
                progress = int((idx / len(df)) * 100)
                progress_callback(progress)

            signal = df.at[idx, 'signal']
            price = df.at[idx, 'close']
            atr = df.at[idx, 'atr']
            exited = False  # Initialize exited flag at the beginning of each iteration

            self.logger.debug(f"Index {idx}: Signal={signal}, Price={price}, Position={position}, Entry={entry_price}, StopLoss={stop_loss_price}, ProfitTarget={profit_target_price}")

            # Handle exiting a LONG position
            if position == 1:
                if signal == TradeSignal.SELL or price <= stop_loss_price or price >= profit_target_price:
                    pnl = price - entry_price
                    equity += pnl
                    df.at[idx, 'exit_price'] = price
                    df.at[idx, 'strategy_returns'] = pnl
                    trades.append({
                        'type': 'LONG',
                        'entry_date': df.at[idx, 'Date'].strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'exit_date': df.at[idx, 'Date'].strftime('%Y-%m-%d'),
                        'exit_price': price,
                        'pnl': pnl
                    })
                    position = 0  # Reset position
                    df.at[idx, 'position'] = position
                    entry_price = 0.0
                    stop_loss_price = 0.0
                    profit_target_price = 0.0
                    exited = True
                    self.logger.debug(f"Exited LONG at index {idx}, price={price}, PnL={pnl:.2f}")
                    continue  # Skip entry logic in this iteration

            # Handle exiting a SHORT position
            elif position == -1:
                if signal == TradeSignal.BUY or price >= stop_loss_price or price <= profit_target_price:
                    pnl = entry_price - price
                    equity += pnl
                    df.at[idx, 'exit_price'] = price
                    df.at[idx, 'strategy_returns'] = pnl
                    trades.append({
                        'type': 'SHORT',
                        'entry_date': df.at[idx, 'Date'].strftime('%Y-%m-%d'),
                        'entry_price': entry_price,
                        'exit_date': df.at[idx, 'Date'].strftime('%Y-%m-%d'),
                        'exit_price': price,
                        'pnl': pnl
                    })
                    position = 0
                    df.at[idx, 'position'] = position
                    entry_price = 0.0
                    stop_loss_price = 0.0
                    profit_target_price = 0.0
                    exited = True
                    self.logger.debug(f"Exited SHORT at index {idx}, price={price}, PnL={pnl:.2f}")
                    continue  # Skip entry logic in this iteration

            # Only enter a new position if no position was exited this iteration
            if position == 0 and not exited:
                # Handle entering a new LONG position
                if signal == TradeSignal.BUY:
                    position = 1
                    entry_price = price
                    stop_loss_price = price - atr * self.config.stop_loss_multiplier
                    profit_target_price = price + price * (self.config.profit_target / 100)
                    df.at[idx, 'entry_price'] = entry_price
                    df.at[idx, 'stop_loss'] = stop_loss_price
                    df.at[idx, 'profit_target'] = profit_target_price
                    df.at[idx, 'position'] = position
                    self.logger.debug(f"Entered LONG at index {idx}, price={price}, StopLoss={stop_loss_price}, ProfitTarget={profit_target_price}")

                # Handle entering a new SHORT position
                elif signal == TradeSignal.SELL:
                    position = -1
                    entry_price = price
                    stop_loss_price = price + atr * self.config.stop_loss_multiplier
                    profit_target_price = price - price * (self.config.profit_target / 100)
                    df.at[idx, 'entry_price'] = entry_price
                    df.at[idx, 'stop_loss'] = stop_loss_price
                    df.at[idx, 'profit_target'] = profit_target_price
                    df.at[idx, 'position'] = position
                    self.logger.debug(f"Entered SHORT at index {idx}, price={price}, StopLoss={stop_loss_price}, ProfitTarget={profit_target_price}")

            # Update equity for the current row
            df.at[idx, 'equity'] = equity

        # Final progress update
        if progress_callback:
            progress_callback(100)

        self.logger.info("Backtest completed.")
        self.logger.info(f"Total Trades: {len(trades)}")
        return df, trades

    def calculate_performance_metrics(self, trades: List[dict], equity_series: pd.Series) -> dict:
        """
        Calculate comprehensive performance metrics.
        """
        self.logger.info("Calculating performance metrics.")

        total_pnl = sum(t['pnl'] for t in trades)
        win_trades = [t for t in trades if t['pnl'] > 0]
        loss_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = (len(win_trades) / len(trades) * 100) if trades else 0
        loss_rate = 100 - win_rate
        avg_pnl = (total_pnl / len(trades)) if trades else 0

        # Calculate maximum drawdown
        cumulative_returns = equity_series - equity_series.iloc[0]
        running_max = cumulative_returns.cummax()
        drawdown = running_max - cumulative_returns
        max_drawdown = drawdown.max()

        # Calculate Sharpe Ratio
        daily_returns = equity_series.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else np.nan

        metrics = {
            'Total PnL': total_pnl,
            'Win Rate (%)': win_rate,
            'Loss Rate (%)': loss_rate,
            'Average PnL per Trade': avg_pnl,
            'Maximum Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }

        self.logger.info("Performance metrics calculated.")
        return metrics

# PyQtGraph Plot Widget with Candlestick, RSI, and MACD Support
class PlotWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create Price Plot
        self.price_plot = self.addPlot(row=0, col=0, title="Price Chart")
        self.price_plot.showGrid(x=True, y=True)
        self.price_plot.setLabel('bottom', 'Date')
        self.price_plot.setLabel('left', 'Price')

        # Create RSI Plot
        self.rsi_plot = self.addPlot(row=1, col=0, title="RSI")
        self.rsi_plot.showGrid(x=True, y=True)
        self.rsi_plot.setLabel('left', 'RSI')
        self.rsi_plot.setYRange(0, 100)

        # Create MACD Plot
        self.macd_plot = self.addPlot(row=2, col=0, title="MACD")
        self.macd_plot.showGrid(x=True, y=True)
        self.macd_plot.setLabel('left', 'MACD')

        # Set X-Link for Synchronized Zoom/Pan
        self.rsi_plot.setXLink(self.price_plot)
        self.macd_plot.setXLink(self.price_plot)

    def plot_candlesticks(self, dates, opens, highs, lows, closes):
        """
        Plot candlestick chart on the price_plot.
        """
        # Clear existing items
        self.price_plot.clear()

        # Create candlesticks
        for i in range(len(dates)):
            if closes[i] >= opens[i]:
                brush = pg.mkBrush('g')  # Green for bullish
                pen = pg.mkPen('g')
            else:
                brush = pg.mkBrush('r')  # Red for bearish
                pen = pg.mkPen('r')
            # Draw the high-low line
            self.price_plot.plot([i, i], [lows[i], highs[i]], pen=pen)
            # Draw the open-close rectangle
            rect = QGraphicsRectItem(i - 0.3, min(opens[i], closes[i]), 0.6, abs(closes[i] - opens[i]))
            rect.setBrush(brush)
            rect.setPen(pen)
            self.price_plot.addItem(rect)

    def add_trade_markers(self, buy_signals, sell_signals):
        """
        Add buy and sell markers to the price_plot.
        """
        # Plot Buy Signals
        if not buy_signals.empty:
            x_buy = buy_signals.index.values
            y_buy = buy_signals['close'].values
            self.price_plot.plot(
                x_buy, y_buy, pen=None, symbol='t', symbolBrush='g',
                symbolSize=15, name='Buy Signal'
            )

        # Plot Sell Signals
        if not sell_signals.empty:
            x_sell = sell_signals.index.values
            y_sell = sell_signals['close'].values
            self.price_plot.plot(
                x_sell, y_sell, pen=None, symbol='x', symbolBrush='r',
                symbolSize=15, name='Sell Signal'
            )

    def plot_indicators(self, df: pd.DataFrame):
        """
        Plot RSI and MACD indicators.
        """
        x = np.arange(len(df))

        # Plot RSI
        self.rsi_plot.clear()
        self.rsi_plot.plot(
            x, df['rsi'], pen=pg.mkPen('magenta', width=1), name='RSI'
        )
        self.rsi_plot.addLine(
            y=70, pen=pg.mkPen('red', width=1, style=Qt.DashLine)
        )
        self.rsi_plot.addLine(
            y=30, pen=pg.mkPen('green', width=1, style=Qt.DashLine)
        )

        # Plot MACD and Signal Line
        self.macd_plot.clear()
        self.macd_plot.plot(
            x, df['macd'], pen=pg.mkPen('blue', width=1), name='MACD'
        )
        self.macd_plot.plot(
            x, df['macd_signal'], pen=pg.mkPen('red', width=1), name='Signal Line'
        )
        # Plot MACD Histogram
        macd_hist = df['macd_diff']
        self.macd_plot.plot(
            x, macd_hist, pen=pg.mkPen('grey', width=1), name='MACD Histogram'
        )

        # Add Legends
        self.price_plot.addLegend()
        self.rsi_plot.addLegend()
        self.macd_plot.addLegend()

        # Enable Auto Range for All Plots
        self.price_plot.enableAutoRange(axis=pg.ViewBox.XYAxes)
        self.rsi_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.macd_plot.enableAutoRange(axis=pg.ViewBox.YAxis)

# Real-Time Trading GUI Class
class RealTimeTradingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Trading")
        self.resize(1600, 1000)
        self.api_connected = False  # Track broker API connection status
        self.api = None  # Alpaca API
        self.strategy = None  # Strategy instance
        self.init_ui()
        self.init_alpaca_api()
        self.portfolio = []
        self.positions = {}
        self.initialize_strategy()

    def init_ui(self):
        """
        Initialize the user interface.
        """
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top Panel: Symbol Input and Market Status
        top_layout = QHBoxLayout()
        symbol_label = QLabel("Stock Symbol:")
        self.symbol_input = QLineEdit("AAPL")
        self.market_status_label = QLabel("Market: Closed")
        top_layout.addWidget(symbol_label)
        top_layout.addWidget(self.symbol_input)
        top_layout.addStretch()
        top_layout.addWidget(self.market_status_label)
        main_layout.addLayout(top_layout)

        # Central Layout: Indicators, Signals, Portfolio
        central_layout = QHBoxLayout()
        main_layout.addLayout(central_layout)

        # Left Panel: Real-Time Indicators
        self.indicator_widget = PlotWidget()
        central_layout.addWidget(self.indicator_widget)

        # Center Panel: Signal Notifications
        signal_layout = QVBoxLayout()
        self.signal_log = QTextEdit()
        self.signal_log.setReadOnly(True)
        signal_layout.addWidget(QLabel("Signal Notifications:"))
        signal_layout.addWidget(self.signal_log)
        central_layout.addLayout(signal_layout)

        # Right Panel: Portfolio Monitoring
        portfolio_layout = QVBoxLayout()
        self.portfolio_table = QTableView()
        portfolio_layout.addWidget(QLabel("Portfolio Monitoring:"))
        portfolio_layout.addWidget(self.portfolio_table)
        central_layout.addLayout(portfolio_layout)

        # Bottom Panel: Trade Execution
        trade_layout = QHBoxLayout()
        self.buy_button = QPushButton("Buy")
        self.buy_button.clicked.connect(self.execute_buy)
        self.sell_button = QPushButton("Sell")
        self.sell_button.clicked.connect(self.execute_sell)
        self.api_status_label = QLabel("Broker: Not Connected")
        trade_layout.addWidget(self.buy_button)
        trade_layout.addWidget(self.sell_button)
        trade_layout.addStretch()
        trade_layout.addWidget(self.api_status_label)
        main_layout.addLayout(trade_layout)

    def init_alpaca_api(self):
        """
        Initialize connection to Alpaca API.
        """
        # Replace these with your actual Alpaca API credentials
        API_KEY = "YOUR_ALPACA_API_KEY"
        SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
        BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading URL

        try:
            self.api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
            account = self.api.get_account()
            if account.status == 'ACTIVE':
                self.api_connected = True
                self.api_status_label.setText("Broker: Connected")
                logger.info("Connected to Alpaca API successfully.")
            else:
                self.api_connected = False
                self.api_status_label.setText("Broker: Inactive")
                logger.warning("Alpaca account is not active.")
        except Exception as e:
            self.api_connected = False
            self.api_status_label.setText("Broker: Connection Failed")
            QMessageBox.critical(self, "API Connection Error", f"Failed to connect to Alpaca API: {e}")
            logger.error(f"Failed to connect to Alpaca API: {e}")

    def initialize_strategy(self):
        """
        Initialize the trading strategy.
        """
        # Initialize StrategyConfig with default or desired parameters
        config = StrategyConfig(
            ema_length=8,
            rsi_length=14,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            macd_fast_window=12,
            macd_slow_window=26,
            macd_signal_window=9,
            adx_window=14,
            vwap_window=14,
            bollinger_window=20,
            atr_window=14,
            risk_percent=0.5,
            profit_target=15.0,
            stop_loss_multiplier=2.0
        )
        self.strategy = Strategy(config=config)
        logger.info("Trading strategy initialized.")

    def execute_buy(self):
        """
        Execute a buy order via Alpaca API.
        """
        if not self.api_connected:
            QMessageBox.warning(self, "API Not Connected", "Cannot execute trade. Alpaca API is not connected.")
            return

        symbol = self.symbol_input.text().strip().upper()
        try:
            # Submit a market buy order for 1 share
            order = self.api.submit_order(
                symbol=symbol,
                qty=1,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            QMessageBox.information(self, "Trade Execution", f"Buy order placed for {symbol}.")
            self.signal_log.append(f"BUY order placed for {symbol} at market price.")
        except Exception as e:
            QMessageBox.critical(self, "Trade Execution Error", f"Failed to place buy order: {e}")
            logger.error(f"Failed to place buy order: {e}")

    def execute_sell(self):
        """
        Execute a sell order via Alpaca API.
        """
        if not self.api_connected:
            QMessageBox.warning(self, "API Not Connected", "Cannot execute trade. Alpaca API is not connected.")
            return

        symbol = self.symbol_input.text().strip().upper()
        try:
            # Submit a market sell order for 1 share
            order = self.api.submit_order(
                symbol=symbol,
                qty=1,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            QMessageBox.information(self, "Trade Execution", f"Sell order placed for {symbol}.")
            self.signal_log.append(f"SELL order placed for {symbol} at market price.")
        except Exception as e:
            QMessageBox.critical(self, "Trade Execution Error", f"Failed to place sell order: {e}")
            logger.error(f"Failed to place sell order: {e}")

    def start_real_time_updates(self):
        """
        Start real-time data updates and portfolio monitoring.
        """
        if not self.api_connected:
            return

        # Start a timer to fetch and update data every minute
        self.timer = QTimer()
        self.timer.timeout.connect(self.fetch_and_update)
        self.timer.start(60000)  # 60,000 ms = 1 minute

        # Fetch initial data
        self.fetch_and_update()

    def fetch_and_update(self):
        """
        Fetch real-time data, update indicators, generate signals, and update portfolio.
        """
        symbol = self.symbol_input.text().strip().upper()
        try:
            # Fetch latest minute data
            barset = self.api.get_barset(symbol, 'minute', limit=100)
            bars = barset[symbol]
            if not bars:
                logger.warning(f"No data fetched for symbol: {symbol}")
                return

            # Convert to DataFrame
            data = [{
                'Date': bar.t,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v
            } for bar in bars]
            df = pd.DataFrame(data)
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Calculate Indicators
            df = self.strategy.calculate_indicators(df)

            # Generate Signals
            df = self.strategy.generate_signals(df)

            # Plot Indicators and Signals
            self.indicator_widget.plot_candlesticks(df['Date'], df['open'], df['high'], df['low'], df['close'])
            buy_signals = df[df['signal'] == TradeSignal.BUY]
            sell_signals = df[df['signal'] == TradeSignal.SELL]
            self.indicator_widget.add_trade_markers(buy_signals, sell_signals)
            self.indicator_widget.plot_indicators(df)

            # Generate and Log Signals
            latest_signal = df.iloc[-1]['signal']
            latest_price = df.iloc[-1]['close']
            timestamp = df.iloc[-1]['Date'].strftime('%Y-%m-%d %H:%M')
            if latest_signal != TradeSignal.HOLD:
                self.signal_log.append(f"{timestamp}: {latest_signal} at ${latest_price:.2f}")

            # Update Portfolio
            self.update_portfolio()

        except Exception as e:
            logger.error(f"Error during real-time update: {e}")

    def update_portfolio(self):
        """
        Fetch and display portfolio positions and PnL.
        """
        try:
            portfolio = self.api.list_positions()
            portfolio_data = []
            for pos in portfolio:
                portfolio_data.append({
                    'Symbol': pos.symbol,
                    'Qty': pos.qty,
                    'Entry Price': pos.avg_entry_price,
                    'Current Price': pos.current_price,
                    'PnL': pos.unrealized_pl
                })
            model = PortfolioTableModel(portfolio_data)
            self.portfolio_table.setModel(model)
            self.portfolio_table.resizeColumnsToContents()
        except Exception as e:
            logger.error(f"Failed to fetch portfolio: {e}")

    def closeEvent(self, event):
        """
        Handle window close event to stop timers and connections gracefully.
        """
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        event.accept()

# Portfolio Table Model for QTableView
class PortfolioTableModel(QAbstractTableModel):
    def __init__(self, portfolio: List[dict]):
        super().__init__()
        self.portfolio = portfolio
        self.headers = ["Symbol", "Qty", "Entry Price", "Current Price", "PnL"]

    def rowCount(self, parent=QModelIndex()):
        return len(self.portfolio)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            trade = self.portfolio[index.row()]
            column = self.headers[index.column()]
            key = column.lower().replace(' ', '_')
            value = trade.get(key, '')
            if isinstance(value, float):
                if column == "PnL":
                    return f"${value:.2f}"
                else:
                    return f"${value:.2f}"
            return str(value)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            else:
                return section + 1
        return None

# Trading Strategy GUI Class
class TradingStrategyGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Strategy Backtester")
        self.resize(1600, 1000)
        self.strategy = None
        self.df = None
        self.backtest_results = None
        self.trades = None
        self.metrics = None

        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface.
        """
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top Layout: Side Panel and Central Plot
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        # Side Panel for Input Parameters
        side_panel = QWidget()
        side_layout = QGridLayout()
        side_panel.setLayout(side_layout)
        side_panel.setFixedWidth(300)  # Adjust width as needed

        # Input Fields

        # Stock Symbol
        symbol_label = QLabel("Stock Symbol:")
        self.symbol_input = QLineEdit("AAPL")
        side_layout.addWidget(symbol_label, 0, 0)
        side_layout.addWidget(self.symbol_input, 0, 1)

        # Date Range
        start_date_label = QLabel("Start Date:")
        self.start_date_input = QDateEdit(calendarPopup=True)
        self.start_date_input.setDate(QDate(2020, 1, 1))
        side_layout.addWidget(start_date_label, 1, 0)
        side_layout.addWidget(self.start_date_input, 1, 1)

        end_date_label = QLabel("End Date:")
        self.end_date_input = QDateEdit(calendarPopup=True)
        self.end_date_input.setDate(QDate.currentDate())
        side_layout.addWidget(end_date_label, 2, 0)
        side_layout.addWidget(self.end_date_input, 2, 1)

        # Strategy Parameters
        params_label = QLabel("Strategy Parameters:")
        params_label.setStyleSheet("font-weight: bold;")
        side_layout.addWidget(params_label, 3, 0, 1, 2)

        # EMA Length
        ema_label = QLabel("EMA Length:")
        self.ema_input = QSpinBox()
        self.ema_input.setRange(1, 100)
        self.ema_input.setValue(8)
        side_layout.addWidget(ema_label, 4, 0)
        side_layout.addWidget(self.ema_input, 4, 1)

        # RSI Length
        rsi_label = QLabel("RSI Length:")
        self.rsi_input = QSpinBox()
        self.rsi_input.setRange(1, 100)
        self.rsi_input.setValue(14)
        side_layout.addWidget(rsi_label, 5, 0)
        side_layout.addWidget(self.rsi_input, 5, 1)

        # RSI Overbought
        rsi_overbought_label = QLabel("RSI Overbought:")
        self.rsi_overbought_input = QDoubleSpinBox()
        self.rsi_overbought_input.setRange(0.0, 100.0)
        self.rsi_overbought_input.setValue(70.0)
        side_layout.addWidget(rsi_overbought_label, 6, 0)
        side_layout.addWidget(self.rsi_overbought_input, 6, 1)

        # RSI Oversold
        rsi_oversold_label = QLabel("RSI Oversold:")
        self.rsi_oversold_input = QDoubleSpinBox()
        self.rsi_oversold_input.setRange(0.0, 100.0)
        self.rsi_oversold_input.setValue(30.0)
        side_layout.addWidget(rsi_oversold_label, 7, 0)
        side_layout.addWidget(self.rsi_oversold_input, 7, 1)

        # MACD Parameters
        macd_label = QLabel("MACD Fast Window:")
        self.macd_fast_input = QSpinBox()
        self.macd_fast_input.setRange(1, 100)
        self.macd_fast_input.setValue(12)
        side_layout.addWidget(macd_label, 8, 0)
        side_layout.addWidget(self.macd_fast_input, 8, 1)

        macd_slow_label = QLabel("MACD Slow Window:")
        self.macd_slow_input = QSpinBox()
        self.macd_slow_input.setRange(1, 100)
        self.macd_slow_input.setValue(26)
        side_layout.addWidget(macd_slow_label, 9, 0)
        side_layout.addWidget(self.macd_slow_input, 9, 1)

        macd_signal_label = QLabel("MACD Signal Window:")
        self.macd_signal_input = QSpinBox()
        self.macd_signal_input.setRange(1, 100)
        self.macd_signal_input.setValue(9)
        side_layout.addWidget(macd_signal_label, 10, 0)
        side_layout.addWidget(self.macd_signal_input, 10, 1)

        # ADX Window
        adx_label = QLabel("ADX Window:")
        self.adx_input = QSpinBox()
        self.adx_input.setRange(1, 100)
        self.adx_input.setValue(14)
        side_layout.addWidget(adx_label, 11, 0)
        side_layout.addWidget(self.adx_input, 11, 1)

        # VWAP Window
        vwap_label = QLabel("VWAP Window:")
        self.vwap_input = QSpinBox()
        self.vwap_input.setRange(1, 100)
        self.vwap_input.setValue(14)
        side_layout.addWidget(vwap_label, 12, 0)
        side_layout.addWidget(self.vwap_input, 12, 1)

        # Bollinger Bands Window
        bollinger_label = QLabel("Bollinger Bands Window:")
        self.bollinger_input = QSpinBox()
        self.bollinger_input.setRange(1, 100)
        self.bollinger_input.setValue(20)
        side_layout.addWidget(bollinger_label, 13, 0)
        side_layout.addWidget(self.bollinger_input, 13, 1)

        # ATR Window
        atr_label = QLabel("ATR Window:")
        self.atr_input = QSpinBox()
        self.atr_input.setRange(1, 100)
        self.atr_input.setValue(14)
        side_layout.addWidget(atr_label, 14, 0)
        side_layout.addWidget(self.atr_input, 14, 1)

        # Risk Percent
        risk_label = QLabel("Risk Percent per Trade:")
        self.risk_input = QDoubleSpinBox()
        self.risk_input.setRange(0.1, 100.0)
        self.risk_input.setValue(0.5)
        side_layout.addWidget(risk_label, 15, 0)
        side_layout.addWidget(self.risk_input, 15, 1)

        # Profit Target
        profit_label = QLabel("Profit Target Percent:")
        self.profit_input = QDoubleSpinBox()
        self.profit_input.setRange(1.0, 100.0)
        self.profit_input.setValue(15.0)
        side_layout.addWidget(profit_label, 16, 0)
        side_layout.addWidget(self.profit_input, 16, 1)

        # Stop Loss Multiplier
        stop_label = QLabel("Stop Loss Multiplier:")
        self.stop_input = QDoubleSpinBox()
        self.stop_input.setRange(1.0, 10.0)
        self.stop_input.setValue(2.0)
        side_layout.addWidget(stop_label, 17, 0)
        side_layout.addWidget(self.stop_input, 17, 1)

        # Spacer to push buttons to the bottom
        side_layout.setRowStretch(18, 1)

        # Button Layout (Run Backtest and Export Results)
        self.run_button = QPushButton("Run Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Status Message Label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: green;")

        # Add Buttons to Side Layout
        side_layout.addWidget(self.run_button, 19, 0, 1, 2)
        side_layout.addWidget(self.export_button, 20, 0, 1, 2)
        side_layout.addWidget(self.progress_bar, 21, 0, 1, 2)
        side_layout.addWidget(self.status_label, 22, 0, 1, 2)

        # Add Side Panel to Top Layout
        top_layout.addWidget(side_panel)

        # Central Panel for Plots
        central_panel = QWidget()
        central_layout = QVBoxLayout()
        central_panel.setLayout(central_layout)

        # Plot Widget using PyQtGraph
        self.plot_widget = PlotWidget()
        central_layout.addWidget(self.plot_widget)

        top_layout.addWidget(central_panel)

        # Bottom Layout: Performance Metrics and Trade Summary
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        # Performance Metrics
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        performance_layout = QVBoxLayout()
        performance_layout.addWidget(QLabel("Performance Metrics:"))
        performance_layout.addWidget(self.results_text)
        bottom_layout.addLayout(performance_layout)

        # Trade Summary Table
        self.trade_table = QTableView()
        table_layout = QVBoxLayout()
        table_layout.addWidget(QLabel("Trade Summary:"))
        table_layout.addWidget(self.trade_table)
        bottom_layout.addLayout(table_layout)

    def run_backtest(self):
        """
        Execute the backtest process.
        """
        symbol = self.symbol_input.text().strip().upper()
        start_date = self.start_date_input.date().toPyDate()
        end_date = self.end_date_input.date().toPyDate()

        # Validate dates
        if start_date >= end_date:
            QMessageBox.warning(self, "Invalid Dates", "Start date must be before end date.")
            return

        # Fetch historical data
        self.status_message("Fetching historical data...", error=False)
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                QMessageBox.warning(self, "No Data", f"No data found for symbol {symbol} in the given date range.")
                self.status_message("No data found.", error=True)
                return
        except Exception as e:
            QMessageBox.critical(self, "Data Fetch Error", f"An error occurred while fetching data: {e}")
            self.status_message("Data fetch error.", error=True)
            return

        # Reset index to get 'Date' as a column
        df.reset_index(inplace=True)

        # Rename columns to match expected names
        rename_columns = {
            'Adj Close': 'adj_close',
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        }

        # Check if 'Open' exists and rename to 'open'
        if 'Open' in df.columns:
            rename_columns['Open'] = 'open'
        else:
            # If 'Open' is missing, fill it with 'close' as a fallback
            df['open'] = df['close']
            logger.warning("'Open' column missing. Filled 'open' with 'close' prices.")

        df.rename(columns=rename_columns, inplace=True)

        # Ensure all required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in required_columns:
            if column not in df.columns:
                if column == 'open':
                    df['open'] = df['close']  # Fallback
                    logger.warning("'open' column missing after renaming. Filled 'open' with 'close' prices.")
                else:
                    QMessageBox.critical(self, "Missing Data", f"Missing required column: {column}")
                    self.status_message(f"Missing column: {column}", error=True)
                    return

        # Initialize StrategyConfig with user inputs
        config = StrategyConfig(
            ema_length=self.ema_input.value(),
            rsi_length=self.rsi_input.value(),
            rsi_overbought=self.rsi_overbought_input.value(),
            rsi_oversold=self.rsi_oversold_input.value(),
            macd_fast_window=self.macd_fast_input.value(),
            macd_slow_window=self.macd_slow_input.value(),
            macd_signal_window=self.macd_signal_input.value(),
            adx_window=self.adx_input.value(),
            vwap_window=self.vwap_input.value(),
            bollinger_window=self.bollinger_input.value(),
            atr_window=self.atr_input.value(),
            risk_percent=self.risk_input.value(),
            profit_target=self.profit_input.value(),
            stop_loss_multiplier=self.stop_input.value()
        )

        # Initialize Strategy
        self.strategy = Strategy(config=config)

        # Calculate Indicators
        self.status_message("Calculating indicators...", error=False)
        try:
            df_with_indicators = self.strategy.calculate_indicators(df)
        except ValueError as e:
            QMessageBox.critical(self, "Indicator Calculation Error", str(e))
            self.status_message("Indicator calculation error.", error=True)
            return

        # Generate Signals
        self.status_message("Generating signals...", error=False)
        df_with_signals = self.strategy.generate_signals(df_with_indicators)

        # Backtest Strategy with Progress Callback
        self.status_message("Running backtest...", error=False)
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress_bar.setValue(0)
        QApplication.processEvents()  # Update UI

        def update_progress(progress):
            self.progress_bar.setValue(progress)
            QApplication.processEvents()

        backtest_results, trades = self.strategy.backtest_strategy(df_with_signals, progress_callback=update_progress)
        self.backtest_results = backtest_results
        self.trades = trades

        # Calculate Performance Metrics
        self.metrics = self.strategy.calculate_performance_metrics(trades, backtest_results['equity'])
        self.display_performance_metrics(self.metrics)

        # Plot Charts
        self.plot_charts(backtest_results)

        # Display Trade Summary
        self.display_trade_summary(trades)

        # Enable Export Button
        self.export_button.setEnabled(True)
        self.run_button.setEnabled(True)

        self.status_message("Backtest completed successfully.", error=False)

    def plot_charts(self, df: pd.DataFrame):
        """
        Plot the candlestick chart along with RSI and MACD indicators.
        """
        # Ensure required columns exist
        required_columns = ['Date', 'open', 'high', 'low', 'close']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

        # Extract data
        dates = df['Date']
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Plot candlesticks
        self.plot_widget.plot_candlesticks(dates, opens, highs, lows, closes)

        # Add Trade Markers
        buy_signals = df[df['signal'] == TradeSignal.BUY]
        sell_signals = df[df['signal'] == TradeSignal.SELL]
        self.plot_widget.add_trade_markers(buy_signals, sell_signals)

        # Plot RSI and MACD Indicators
        self.plot_widget.plot_indicators(df)

    def display_performance_metrics(self, metrics: dict):
        """
        Display performance metrics in the QTextEdit widget.
        """
        summary = (
            f"Total PnL: ${metrics['Total PnL']:.2f}\n"
            f"Win Rate: {metrics['Win Rate (%)']:.2f}%\n"
            f"Loss Rate: {metrics['Loss Rate (%)']:.2f}%\n"
            f"Average PnL per Trade: ${metrics['Average PnL per Trade']:.2f}\n"
            f"Maximum Drawdown: ${metrics['Maximum Drawdown']:.2f}\n"
            f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}\n"
        )
        self.results_text.setText(summary)

    def display_trade_summary(self, trades: List[dict]):
        """
        Display trade summary in the QTableView.
        """
        if not trades:
            self.trade_table.setModel(None)
            QMessageBox.information(self, "No Trades", "No trades were executed.")
            return

        model = TradeTableModel(trades)
        self.trade_table.setModel(model)
        self.trade_table.resizeColumnsToContents()

    def export_results(self):
        """
        Export backtest results and trade summary to CSV files.
        """
        if self.backtest_results is None or self.trades is None:
            QMessageBox.warning(self, "No Data", "No backtest results to export.")
            return

        # Choose file location for backtest results
        options = QFileDialog.Options()
        backtest_path, _ = QFileDialog.getSaveFileName(
            self, "Save Backtest Results", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if backtest_path:
            try:
                self.backtest_results.to_csv(backtest_path, index=False)
                QMessageBox.information(self, "Export Successful", f"Backtest results saved to {backtest_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred while saving the backtest results: {e}")

        # Choose file location for trade summary
        trade_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trade Summary", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if trade_path:
            try:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(trade_path, index=False)
                QMessageBox.information(self, "Export Successful", f"Trade summary saved to {trade_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred while saving the trade summary: {e}")

    def status_message(self, message: str, error: bool = False):
        """
        Update the status message label with the provided message.
        """
        if error:
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: green;")
        self.status_label.setText(message)

# Main Application Class with Tab Support
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Strategy Backtester Application")
        self.resize(1800, 1200)

        # Initialize Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add Trading Backtester as a Tab
        self.backtester_tab = TradingStrategyGUI()
        self.tabs.addTab(self.backtester_tab, "Backtester")

        # Add Real-Time Trading as a Tab
        self.real_time_trading_tab = RealTimeTradingGUI()
        self.tabs.addTab(self.real_time_trading_tab, "Real-Time Trading")

# Main Function
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Start real-time updates in the Real-Time Trading Tab
    window.real_time_trading_tab.start_real_time_updates()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
