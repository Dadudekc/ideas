from typing import List
from basicbot.trading_api import TradingAPI
import logging

class Trader:
    """
    Trader class to execute trades based on signals.
    """
    def __init__(self, api: TradingAPI, logger: logging.Logger):
        self.api = api
        self.logger = logger
        self.position = 0  # 1 for long, -1 for short, 0 for no position

    def execute_signals(self, signals: List[str], symbol: str, current_price: float, qty: int):
        """
        Execute trades based on generated signals.
        """
        for signal in signals:
            if signal == 'BUY' and self.position != 1:
                self.buy(symbol, qty, current_price)
            elif signal == 'SELL' and self.position != -1:
                self.sell(symbol, qty, current_price)
            elif signal == 'HOLD':
                self.logger.info("Holding position.")
            else:
                self.logger.info("No action taken.")

    def buy(self, symbol: str, qty: int, price: float):
        self.position = 1
        self.logger.info(f"Executing BUY for {symbol} at {price}.")
        self.api.place_order(symbol=symbol, qty=qty, side="buy")

    def sell(self, symbol: str, qty: int, price: float):
        self.position = -1
        self.logger.info(f"Executing SELL for {symbol} at {price}.")
        self.api.place_order(symbol=symbol, qty=qty, side="sell")
