import logging
import robin_stocks.robinhood as rh
import pandas as pd
import csv
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict

# Load configuration from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class RobinhoodClient:
    """Handles authentication and data retrieval from Robinhood."""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.authenticated = self.login()

    def login(self) -> bool:
        """Logs into Robinhood using provided credentials."""
        logging.info("Attempting to log in to Robinhood...")
        try:
            rh.login(self.username, self.password)
            logging.info("Login successful.")
            return True
        except Exception as e:
            logging.error(f"Login failed: {e}")
            return False

    def get_options_trade_history(self) -> Optional[List[Dict]]:
        """Fetches all options trade data (open and closed) from Robinhood."""
        if not self.authenticated:
            logging.error("Not authenticated. Cannot retrieve options trade history.")
            return None
        logging.info("Retrieving all options trade positions...")
        try:
            trade_history = rh.options.get_all_option_positions()
            if trade_history:
                logging.info("Options trade history retrieved successfully.")
                return trade_history
            else:
                logging.warning("No trade history found.")
                return None
        except Exception as e:
            logging.error(f"Failed to retrieve trade history: {e}")
            return None

    def get_historical_stock_data(self, symbol: str, interval: str = "day", span: str = "year") -> pd.DataFrame:
        """Fetches historical stock data and returns it as a DataFrame."""
        logging.info(f"Fetching historical data for {symbol}...")
        try:
            data = rh.stocks.get_stock_historicals(symbol, interval=interval, span=span)
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['begins_at'])
            for price_field in ['open_price', 'close_price', 'high_price', 'low_price']:
                df[price_field] = df[price_field].astype(float)
            df['volume'] = df['volume'].astype(int)
            return df[['timestamp', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
        except Exception as e:
            logging.error(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()

    def get_options_data(self, symbol: str, expiration_date: str, option_type: str = "call") -> pd.DataFrame:
        """Fetches options data for a specific expiration date and type, handling missing fields dynamically."""
        logging.info(f"Fetching {option_type} options data for {symbol} on {expiration_date}...")
        try:
            options = rh.options.find_tradable_options(symbol)
            
            # Convert to DataFrame and filter by expiration date and option type
            df = pd.DataFrame(options)
            
            # Log available columns for debugging purposes
            logging.debug(f"Available columns in options data for {symbol}: {df.columns.tolist()}")

            # Apply expiration date and option type filters
            df = df[(df['expiration_date'] == expiration_date) & (df['type'] == option_type)]
            
            # Dynamically select columns based on availability in the DataFrame
            export_columns = [col for col in df.columns if col in ['expiration_date', 'strike_price', 'ask_price', 'bid_price', 'mark_price']]
            
            # If only non-essential fields are missing, proceed with export
            if export_columns:
                for col in export_columns:
                    if col in ['strike_price', 'ask_price', 'bid_price', 'mark_price']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float where possible
                
                # Export only available fields
                logging.info(f"Available data columns for export: {export_columns}")
                return df[export_columns].dropna(how='all')
            else:
                logging.warning(f"No essential columns available to export for {symbol} on {expiration_date}.")
                return pd.DataFrame()  # Return an empty DataFrame if no essential columns are available
        except Exception as e:
            logging.error(f"Error fetching options data for {symbol}: {e}")
            return pd.DataFrame()


class DataExporter:
    """Exports data to CSV files."""

    @staticmethod
    def export_trade_history(trade_history: List[Dict], filename: str = "all_options_trade_history.csv") -> None:
        """Exports trade history to a CSV file."""
        if not trade_history:
            logging.warning("No trade history to export.")
            return
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=trade_history[0].keys())
                writer.writeheader()
                writer.writerows(trade_history)
            logging.info(f"Trade history successfully exported to {filename}.")
        except Exception as e:
            logging.error(f"Failed to export trade history to CSV: {e}")

    @staticmethod
    def export_dataframe_to_csv(df: pd.DataFrame, filename: str) -> None:
        """Exports a DataFrame to a CSV file."""
        if df.empty:
            logging.warning(f"No data to export for {filename}.")
            return
        try:
            df.to_csv(filename, index=False)
            logging.info(f"Data successfully exported to {filename}.")
        except Exception as e:
            logging.error(f"Failed to export data to {filename}: {e}")


def main() -> None:
    """Main function to handle data retrieval and export."""
    username = os.getenv("ROBINHOOD_USERNAME")
    password = os.getenv("ROBINHOOD_PASSWORD")
    symbols = ["AAPL", "TSLA"]
    expiration_dates = ["2023-12-15", "2024-01-19"]

    if not username or not password:
        logging.error("Username or password not set. Please configure them in the .env file.")
        return

    # Initialize Robinhood Client and Data Exporter
    client = RobinhoodClient(username, password)
    exporter = DataExporter()

    # Retrieve and export options trade history
    trade_history = client.get_options_trade_history()
    if trade_history:
        exporter.export_trade_history(trade_history)

    # Download and export historical stock data
    for symbol in symbols:
        stock_data = client.get_historical_stock_data(symbol)
        exporter.export_dataframe_to_csv(stock_data, f"{symbol}_historical_data.csv")

    # Download and export options data
    for symbol in symbols:
        for expiration_date in expiration_dates:
            for option_type in ["call", "put"]:
                options_data = client.get_options_data(symbol, expiration_date, option_type)
                filename = f"{symbol}_{expiration_date}_{option_type}_options_data.csv"
                exporter.export_dataframe_to_csv(options_data, filename)

# Run the script if executed directly
if __name__ == "__main__":
    main()
