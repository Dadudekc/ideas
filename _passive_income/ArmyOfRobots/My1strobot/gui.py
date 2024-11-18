# gui.py

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QComboBox,
    QMessageBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
import sys
import alpaca_trade_api as tradeapi  # Correct import for Alpaca's tradeapi

from utils import load_config, save_config, setup_logger
from portfolio import Portfolio
from strategy import Strategy
from backtester import Backtester
from paper_trader import PaperTrader
from threads import BacktestThread, PaperTradeThread

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
            self.backtest_thread.finished.connect(self.backtest_finished)  # Connect the finished signal
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

    def backtest_finished(self):
        """
        Slot to handle the completion of the backtest thread.
        """
        self.log_text.append("[LOG] Backtest completed.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.backtest_thread = None  # Clean up the thread reference

    def stop_trading(self):
        mode = self.mode_combo.currentText()
        if mode in ["paper_trade", "live_trade"] and self.paper_trade_thread:
            self.paper_trade_thread.trader.stop_trading()
            self.paper_trade_thread.quit()
            self.paper_trade_thread.wait()
            self.log_text.append("[LOG] Trading session stopped.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.paper_trade_thread = None  # Clean up the thread reference
        else:
            QMessageBox.warning(self, "Stop Error", "No active trading session to stop.")

    def update_log(self, message):
        self.log_text.append(message)
        # Update portfolio status
        self.balance_label.setText(f"Balance: ${self.portfolio.balance:.2f}")
        self.positions_label.setText(f"Positions: {self.portfolio.positions}")
