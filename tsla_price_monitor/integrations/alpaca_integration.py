# integrations/alpaca_integration.py
from alpaca_trade_api.rest import REST, TimeFrame
from typing import Optional, Dict, Any
from utils.logger import logger
from config import Config
import os

class AlpacaIntegration:
    """
    Handles interaction with the Alpaca API for trading functionalities.
    """
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL")
        logger.info(f"Using Alpaca base URL: '{self.base_url}'")  # Debug log for base URL

        # Check if URL and keys are initialized correctly
        if not self.api_key or not self.secret_key or not self.base_url:
            logger.error("Alpaca API keys or URL are missing. Check environment variables.")
            return  # Exit initialization if keys or URL are missing
        
        try:
            self.api = REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca API. Account status: {account.status}")
        except Exception as e:
            logger.error(f"Error connecting to Alpaca API: {e}")


    def fetch_account_info(self) -> Optional[Dict[str, Any]]:
        """Fetches account information from Alpaca."""
        if not self.api:
            logger.error("Alpaca API is not initialized.")
            return None
        try:
            account = self.api.get_account()
            account_info = {
                "status": account.status,
                "cash": account.cash,
                "portfolio_value": account.portfolio_value,
                "last_equity": account.last_equity
            }
            logger.info("Fetched account information from Alpaca.")
            return account_info
        except Exception as e:
            logger.error(f"Error fetching account info from Alpaca: {e}")
            return None

    def fetch_stock_data(self, symbol: str = "TSLA") -> Optional[Dict[str, Any]]:
        """Fetches recent stock data from Alpaca."""
        if not self.api:
            logger.error("Alpaca API is not initialized.")
            return None
        try:
            barset = self.api.get_bars(symbol, TimeFrame.Day, limit=5).df
            if barset.empty:
                logger.warning(f"No data fetched for {symbol} from Alpaca.")
                return None
            latest = barset.iloc[-1]
            stock_data = {
                "symbol": symbol,
                "open": latest['open'],
                "high": latest['high'],
                "low": latest['low'],
                "close": latest['close'],
                "volume": latest['volume'],
                "time": latest.name.strftime("%Y-%m-%d")
            }
            logger.info(f"Fetched stock data from Alpaca: {stock_data}")
            return stock_data
        except Exception as e:
            logger.error(f"Error fetching stock data from Alpaca: {e}")
            return None

    def place_order(self, symbol: str, qty: int, side: str, type_: str, time_in_force: str) -> Optional[Any]:
        """Places an order via Alpaca."""
        if not self.api:
            logger.error("Alpaca API is not initialized.")
            return None
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type_,
                time_in_force=time_in_force
            )
            logger.info(f"Placed order: {order}")
            return order
        except Exception as e:
            logger.error(f"Error placing order via Alpaca: {e}")
            return None
