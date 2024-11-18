import unittest
from unittest.mock import MagicMock
from basicbot.trader import Trader
from basicbot.trading_api import TradingAPI
import logging

class TestTraderAPI(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger("TestTraderAPI")
        logger.setLevel(logging.INFO)
        self.mock_api = MagicMock(spec=TradingAPI)
        self.trader = Trader(api=self.mock_api, logger=logger)

    def test_buy_order(self):
        self.trader.buy(symbol="AAPL", qty=10, price=150.0)
        self.mock_api.place_order.assert_called_once_with(symbol="AAPL", qty=10, side="buy")

    def test_sell_order(self):
        self.trader.sell(symbol="AAPL", qty=5, price=145.0)
        self.mock_api.place_order.assert_called_once_with(symbol="AAPL", qty=5, side="sell")

    def test_execute_signals(self):
        signals = ["BUY", "HOLD", "SELL"]
        self.trader.execute_signals(signals, symbol="AAPL", current_price=150.0, qty=10)
        self.mock_api.place_order.assert_any_call(symbol="AAPL", qty=10, side="buy")
        self.mock_api.place_order.assert_any_call(symbol="AAPL", qty=10, side="sell")

if __name__ == "__main__":
    unittest.main()
