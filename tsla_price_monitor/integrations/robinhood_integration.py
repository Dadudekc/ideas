# integrations/robinhood_integration.py
import robin_stocks.robinhood as r
import pandas as pd
from typing import Optional, Dict, Any
from utils.logger import logger
from config import Config
from getpass import getpass

class RobinhoodIntegration:
    """
    Handles interaction with the Robinhood API for trading functionalities.
    """
    def __init__(self):
        self.username = Config.ROBINHOOD_USERNAME
        self.password = Config.ROBINHOOD_PASSWORD
        self.logged_in = False
        self.login()

    def login(self):
        """Logs into Robinhood using credentials."""
        if not self.username:
            self.username = input("Enter your Robinhood username: ")
        if not self.password:
            self.password = getpass("Enter your Robinhood password: ")
        try:
            login = r.login(self.username, self.password)
            if login:
                self.logged_in = True
                logger.info("Successfully logged into Robinhood.")
            else:
                logger.error("Failed to log into Robinhood.")
        except Exception as e:
            logger.error(f"Error logging into Robinhood: {e}")

    def fetch_options_trading_history(self) -> Optional[pd.DataFrame]:
        """Fetches detailed options trading history from Robinhood."""
        if not self.logged_in:
            logger.error("Not logged into Robinhood.")
            return None
        try:
            options_orders = r.options.get_all_option_orders()
            options_df = pd.DataFrame(options_orders)
            # Ensure numeric data where applicable
            numeric_columns = ['quantity', 'average_price', 'trade_value_multiplier']
            for col in numeric_columns:
                options_df[col] = pd.to_numeric(options_df[col], errors='coerce')
            logger.info("Fetched options trading history from Robinhood.")
            return options_df
        except Exception as e:
            logger.error(f"Error fetching options trading history: {e}")
            return None

    def calculate_metrics(self, df: pd.DataFrame, trade_type: str) -> Dict[str, Any]:
        """Calculate trading metrics for either stock or options trading data."""
        try:
            df['value'] = df['quantity'] * df['average_price']
            if trade_type == 'stock':
                df['profit_loss'] = df.apply(lambda row: row['executed_notional'] if row['side'] == 'sell' else -row['executed_notional'], axis=1)
            elif trade_type == 'option':
                df['profit_loss'] = df.apply(lambda row: row['quantity'] * row['average_price'] * row['trade_value_multiplier'] * (1 if row['type'] == 'sell' else -1), axis=1)

            total_profit_loss = df['profit_loss'].sum()
            wins = df[df['profit_loss'] > 0]
            losses = df[df['profit_loss'] < 0]
            metrics = {
                "total_profit_loss": total_profit_loss,
                "win_rate": (len(wins) / len(df) * 100) if len(df) > 0 else 0,
                "avg_profit": wins['profit_loss'].mean() if not wins.empty else 0,
                "avg_loss": losses['profit_loss'].mean() if not losses.empty else 0
            }
            logger.info(f"Calculated {trade_type} trading metrics.")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
