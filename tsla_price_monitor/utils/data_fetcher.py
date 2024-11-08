# utils/data_fetcher.py
import yfinance as yf
from typing import Optional, Dict, Any
from utils.logger import logger

class DataFetcher:
    """
    Fetches daily price data for TSLA using yfinance.
    """
    def __init__(self, symbol: str = "TSLA"):
        self.symbol = symbol

    def fetch_daily_price(self) -> Optional[Dict[str, Any]]:
        """Fetches the latest daily price data for TSLA."""
        try:
            stock = yf.Ticker(self.symbol)
            data = stock.history(period="1d")
            if not data.empty:
                latest = data.iloc[-1]
                price_info = {
                    "date": latest.name.strftime("%Y-%m-%d"),
                    "open": latest['Open'],
                    "high": latest['High'],
                    "low": latest['Low'],
                    "close": latest['Close'],
                    "volume": latest['Volume']
                }
                logger.info(f"Fetched daily price for {self.symbol}: {price_info}")
                return price_info
            else:
                logger.warning(f"No data fetched for {self.symbol}.")
                return None
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {e}")
            return None
