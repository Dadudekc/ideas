import unittest
from unittest.mock import patch, MagicMock
from robinhood_options_history import login, get_trade_history, display_trade_history, logout

class TestRobinhoodOptionsHistory(unittest.TestCase):

    @patch("robinhood_options_history.rh.login")
    def test_login_successful(self, mock_login):
        """Test successful login without MFA"""
        mock_login.return_value = {"detail": ""}
        try:
            login("test_user", "test_pass")
            mock_login.assert_called_once_with("test_user", "test_pass")
        except Exception as e:
            self.fail(f"login() raised an exception unexpectedly: {e}")

    @patch("robinhood_options_history.rh.login")
    @patch("robinhood_options_history.input", return_value="123456")
    def test_login_with_mfa(self, mock_input, mock_login):
        """Test login with MFA required"""
        mock_login.side_effect = [
            {"detail": "mfa_required"},  # First call prompts for MFA
            {"detail": ""}  # Second call succeeds
        ]
        try:
            login("test_user", "test_pass")
            self.assertEqual(mock_login.call_count, 2)
            mock_login.assert_any_call("test_user", "test_pass")
        except Exception as e:
            self.fail(f"login() raised an exception with MFA unexpectedly: {e}")

    @patch("robinhood_options_history.rh.options.get_all_option_orders")
    def test_get_trade_history_successful(self, mock_get_all_option_orders):
        """Test successful retrieval of trade history"""
        mock_trade_data = [{"chain_symbol": "AAPL", "id": "123", "quantity": "1"}]
        mock_get_all_option_orders.return_value = mock_trade_data
        result = get_trade_history()
        mock_get_all_option_orders.assert_called_once()
        self.assertEqual(result, mock_trade_data)

    @patch("robinhood_options_history.rh.options.get_all_option_orders")
    def test_get_trade_history_network_error(self, mock_get_all_option_orders):
        """Test trade history retrieval with a network error and retry"""
        mock_get_all_option_orders.side_effect = ConnectionError("Network failure")
        with self.assertRaises(Exception) as context:
            get_trade_history(retries=2, delay=0)
        self.assertTrue("Failed to retrieve trade history after multiple attempts" in str(context.exception))

    @patch("builtins.print")
    def test_display_trade_history_empty(self, mock_print):
        """Test display with empty trade history"""
        display_trade_history([])
        mock_print.assert_called_once_with("No trades to display.")

    @patch("builtins.print")
    def test_display_trade_history(self, mock_print):
        """Test display with sample trade history"""
        mock_trade_data = [{"chain_symbol": "AAPL", "id": "123", "quantity": "1"}]
        display_trade_history(mock_trade_data)
        mock_print.assert_any_call("Trade Details:")
        mock_print.assert_any_call("  Symbol: AAPL")
        mock_print.assert_any_call("  Order ID: 123")
        mock_print.assert_any_call("  Quantity: 1")

    @patch("robinhood_options_history.rh.logout")
    def test_logout_successful(self, mock_logout):
        """Test successful logout"""
        try:
            logout()
            mock_logout.assert_called_once()
        except Exception as e:
            self.fail(f"logout() raised an exception unexpectedly: {e}")

    @patch("robinhood_options_history.rh.logout", side_effect=Exception("Logout error"))
    def test_logout_with_error(self, mock_logout):
        """Test logout with an error handling"""
        with self.assertLogs(level='ERROR') as log:
            logout()
            self.assertIn("ERROR:Error during logout: Logout error", log.output)

if __name__ == "__main__":
    unittest.main()
