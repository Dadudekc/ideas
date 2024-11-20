import unittest
from unittest.mock import patch, Mock
import requests  # Ensure requests is imported
from basicbot.trading_api import TradingAPI
import logging


class TestTradingAPI(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        self.api_key = "test_api_key"
        self.api_secret = "test_api_secret"
        self.base_url = "https://paper-api.alpaca.markets"
        self.logger = logging.getLogger("TestTradingAPI")
        self.logger.setLevel(logging.CRITICAL)  # Suppress logs during tests
        self.trading_api = TradingAPI(self.api_key, self.api_secret, self.base_url, self.logger)

    @patch("requests.Session.post")
    def test_place_order_success(self, mock_post):
        """
        Test successful order placement.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "order123", "status": "new"}
        mock_post.return_value = mock_response

        result = self.trading_api.place_order(
            symbol="AAPL", qty=10, side="buy", type_="market", time_in_force="gtc"
        )

        mock_post.assert_called_once_with(
            f"{self.base_url}/v2/orders",
            json={
                "symbol": "AAPL",
                "qty": 10,
                "side": "buy",
                "type": "market",
                "time_in_force": "gtc"
            }
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "order123")
        self.assertEqual(result["status"], "new")

    @patch("requests.Session.post")
    def test_place_order_failure(self, mock_post):
        """
        Test failed order placement due to an API error.
        """
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.RequestException("Bad Request")
        mock_post.return_value = mock_response

        result = self.trading_api.place_order(
            symbol="AAPL", qty=10, side="buy", type_="market", time_in_force="gtc"
        )

        mock_post.assert_called_once()
        self.assertIsNone(result)

    @patch("requests.Session.get")
    def test_get_account_success(self, mock_get):
        """
        Test successful account details retrieval.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"account_id": "account123", "status": "ACTIVE"}
        mock_get.return_value = mock_response

        result = self.trading_api.get_account()

        mock_get.assert_called_once_with(f"{self.base_url}/v2/account")
        self.assertIsNotNone(result)
        self.assertEqual(result["account_id"], "account123")
        self.assertEqual(result["status"], "ACTIVE")

    @patch("requests.Session.get")
    def test_get_account_failure(self, mock_get):
        """
        Test failed account details retrieval due to an API error.
        """
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.RequestException("Forbidden")
        mock_get.return_value = mock_response

        result = self.trading_api.get_account()

        mock_get.assert_called_once()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
