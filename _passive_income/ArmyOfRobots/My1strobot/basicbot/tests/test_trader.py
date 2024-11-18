# test_trader.py

import unittest
import logging
from basicbot.trader import Trader
from unittest.mock import MagicMock

class TestTrader(unittest.TestCase):
    def setUp(self):
        # Set up a logger
        self.logger = logging.getLogger("TestTrader")
        self.logger.setLevel(logging.INFO)

        # Mock the TradingAPI
        self.mock_api = MagicMock()

        # Initialize Trader with the mocked API and logger
        self.trader = Trader(api=self.mock_api, logger=self.logger)

    def test_buy_signal(self):
        """
        Test executing a BUY signal.
        """
        self.trader.execute_signals(["BUY"], symbol="AAPL", current_price=150.0, qty=10)
        self.mock_api.place_order.assert_called_once_with(symbol="AAPL", qty=10, side="buy")

    def test_sell_signal(self):
        """
        Test executing a SELL signal.
        """
        self.trader.execute_signals(["SELL"], symbol="AAPL", current_price=150.0, qty=5)
        self.mock_api.place_order.assert_called_once_with(symbol="AAPL", qty=5, side="sell")

    def test_hold_signal(self):
        """
        Test executing a HOLD signal.
        """
        self.trader.execute_signals(["HOLD"], symbol="AAPL", current_price=150.0, qty=0)
        self.mock_api.place_order.assert_not_called()

    def test_invalid_signal(self):
        """
        Test handling an invalid signal.
        """
        self.trader.execute_signals(["INVALID"], symbol="AAPL", current_price=150.0, qty=0)
        self.mock_api.place_order.assert_not_called()

    def test_mixed_signals(self):
        """
        Test executing a mix of signals.
        """
        self.trader.execute_signals(["BUY", "HOLD", "SELL"], symbol="AAPL", current_price=150.0, qty=10)
        self.mock_api.place_order.assert_any_call(symbol="AAPL", qty=10, side="buy")
        self.mock_api.place_order.assert_any_call(symbol="AAPL", qty=10, side="sell")

if __name__ == "__main__":
    unittest.main()