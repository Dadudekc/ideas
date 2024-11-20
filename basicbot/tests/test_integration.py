import unittest
from unittest.mock import patch, Mock
from pathlib import Path
from basicbot.config import load_config
from basicbot.strategy import Strategy, StrategyConfig, TradeSignal
from basicbot.trading_api import TradingAPI
from basicbot.notifier import Notifier
from basicbot.data_fetcher import DataFetchUtils
import pandas as pd

class TestTradingWorkflow(unittest.TestCase):
    def setUp(self):
        self.config_path = "temp_config.yaml"
        self.config = load_config(self.config_path)
        self.strategy_config = StrategyConfig()
        self.strategy = Strategy(config=self.strategy_config)
        self.logger = Mock()  # Mock logger for testing
        self.data_fetcher = DataFetchUtils(logger=self.logger, config=self.config)
        self.api = TradingAPI("dummy_key", "dummy_secret", "https://fake.api", self.logger)
        self.notifier = Notifier(
            {
                "smtp_server": "smtp.fake.com",
                "smtp_port": 587,
                "username": "user@fake.com",
                "password": "password",
                "from_addr": "user@fake.com",
                "to_addr": "admin@fake.com",
            },
            self.logger,
        )

    @patch("basicbot.data_fetcher.DataFetchUtils.fetch_stock_data_async")
    @patch("basicbot.trading_api.TradingAPI.place_order")
    @patch("basicbot.notifier.Notifier.send_email")
    def test_trading_workflow(self, mock_send_email, mock_place_order, mock_fetch_data):
        """
        Test a full trading workflow from data fetching to order placement and notification.
        """
        # Mock data fetching with adjusted data for both BUY and SELL conditions
        mock_fetch_data.return_value = pd.DataFrame(
            {
                "date": ["2023-10-31", "2023-11-01", "2023-11-02"],
                "close": [148.0, 151.0, 140.0],          # Adjusted for BUY and SELL
                "high": [150.0, 152.0, 143.0],
                "low": [146.0, 148.0, 138.0],
                "vwap": [149.0, 148.0, 141.0],          # Adjusted for BUY and SELL
                "ema": [147.5, 150.0, 141.5],           # Adjusted for BUY and SELL
                "rsi": [35.0, 40.0, 60.0],              # BUY: 40.0, SELL: 60.0
                "macd_diff": [-0.5, 0.1, -0.5],         # BUY: 0.1, SELL: -0.5
                "adx": [22.0, 25.0, 22.0],              # All days > 20
                "volume": [800000, 1000000, 1200000],   # All days > volume_sma_20
                "volume_sma_20": [900000, 900000, 1100000],  # Volume SMA values
            }
        )

        # Mock order placement
        mock_place_order.return_value = {"status": "success", "order_id": "12345"}

        # Mock email sending
        mock_send_email.return_value = None

        # Simulate data fetching
        data = self.data_fetcher.fetch_stock_data_async("TSLA", "2023-11-01", "2023-11-02", "D")
        signals = self.strategy.generate_signals(data)

        # Debugging: Print the generated signals DataFrame
        print("Generated Signals DataFrame:")
        print(signals)

        # Assert that 'BUY' and 'SELL' signals are present
        self.assertIn("BUY", signals["signal"].unique(), "Expected 'BUY' signal not found.")
        self.assertIn("SELL", signals["signal"].unique(), "Expected 'SELL' signal not found.")

        # Verify specific rows for 'BUY' and 'SELL'
        buy_signals = signals[signals["signal"] == "BUY"]
        sell_signals = signals[signals["signal"] == "SELL"]

        print("BUY Signals:")
        print(buy_signals)
        print("SELL Signals:")
        print(sell_signals)

        self.assertEqual(len(buy_signals), 1, "There should be exactly one 'BUY' signal.")
        self.assertEqual(len(sell_signals), 1, "There should be exactly one 'SELL' signal.")
        self.assertEqual(buy_signals.iloc[0]["date"], "2023-11-01", "BUY signal should be on 2023-11-01.")
        self.assertEqual(sell_signals.iloc[0]["date"], "2023-11-02", "SELL signal should be on 2023-11-02.")

        # Simulate order placement
        order_response = self.api.place_order(symbol="TSLA", qty=10, side="BUY")
        self.assertEqual(order_response["status"], "success")

        # Simulate email notification
        self.notifier.send_email(subject="Trade Executed", message="Order 12345 was placed.")
        mock_send_email.assert_called_once()

    def tearDown(self):
        # Clean up temporary configurations
        config_path = Path(self.config_path)
        if config_path.exists():
            config_path.unlink()

if __name__ == "__main__":
    unittest.main()
