import requests
import logging

class TradingAPI:
    """
    Trading API integration for real-time trading.
    """
    def __init__(self, api_key: str, api_secret: str, base_url: str, logger: logging.Logger):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        })

    def place_order(self, symbol: str, qty: int, side: str, type_: str = "market", time_in_force: str = "gtc"):
        """
        Place a live order.
        """
        endpoint = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type_,
            "time_in_force": time_in_force
        }
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            self.logger.info(f"Order placed: {response.json()}")
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    def get_account(self):
        """
        Get account details.
        """
        endpoint = f"{self.base_url}/v2/account"
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            self.logger.info(f"Account details fetched: {response.json()}")
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching account details: {e}")
            return None
