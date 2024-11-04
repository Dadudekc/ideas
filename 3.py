import logging
import robin_stocks.robinhood as rh
import csv
import os
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load configuration from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def login(username: str, password: str) -> bool:
    """Logs into Robinhood using the provided credentials."""
    logging.info("Attempting to log in to Robinhood...")
    try:
        rh.login(username, password)
        logging.info("Login successful.")
        return True
    except Exception as e:
        logging.error(f"Login failed: {e}")
        return False

def get_all_options_trade_history() -> Optional[List[Dict]]:
    """Fetches all options trade data (open and closed) from Robinhood."""
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

def parse_trade_data(raw_data: str) -> List[Dict[str, str]]:
    """
    Parses raw trade data provided as a string, where each line is a trade entry.
    Returns a list of dictionaries, each representing a trade with parsed details.
    """
    headers = [
        "account_url", "account_id", "quantity", "order_id", "symbol", "position_id", 
        "option_instrument_url", "side", "average_price", "pending_quantity", "quantity_available",
        "strike_price", "adjusted_mark_price", "adjusted_cost", "realized_intraday_pl", 
        "realized_intraday_cost", "realized_intraday_plpc", "created_at", "expiration_date", 
        "multiplier", "updated_at", "position_url", "instrument_id", "position_quantity", 
        "pending_sell_quantity", "type", "intraday_multiplier", "maintenance_ratio", 
        "multiplier_type"
    ]

    data_lines = [line.strip() for line in raw_data.strip().splitlines() if line.strip()]
    parsed_data = [dict(zip(headers, line.split(','))) for line in data_lines]
    
    logging.info(f"Parsed {len(parsed_data)} trade entries.")
    return parsed_data

def display_trade_history(trade_history: List[Dict]) -> None:
    """Displays the trade history in a readable format."""
    if not trade_history:
        logging.info("No trade history to display.")
        return
    
    for trade in trade_history:
        print("\nTrade Details:")
        print(f"  Symbol: {trade.get('symbol', 'N/A')}")
        print(f"  Order ID: {trade.get('order_id', 'N/A')}")
        print(f"  Quantity: {trade.get('quantity', 'N/A')}")
        print(f"  Side: {trade.get('side', 'N/A')}")
        print(f"  Price: {trade.get('average_price', 'N/A')}")
        print(f"  Status: {trade.get('type', 'N/A')}")
        print(f"  Date: {trade.get('created_at', 'N/A')}")
        print(f"  Expiration Date: {trade.get('expiration_date', 'N/A')}")
        print(f"  Strike Price: {trade.get('strike_price', 'N/A')}")
        print(f"  Option Type: {trade.get('type', 'N/A')}")

def export_to_csv(trade_history: List[Dict], filename: str = "all_options_trade_history.csv") -> None:
    """Exports the trade history to a CSV file."""
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

def main() -> None:
    """Main function to handle login, retrieval, display, and export of trade history."""
    username = os.getenv("ROBINHOOD_USERNAME")
    password = os.getenv("ROBINHOOD_PASSWORD")

    if not username or not password:
        logging.error("Username or password not set. Please configure them in the .env file.")
        return

    if login(username, password):
        # Retrieve raw trade data (Example: Simulated data read as a single string from a file or API response)
        raw_trade_data = """
        https://api.robinhood.com/accounts/5SE25445/,5SE25445,0.0000,cd15fc05-93fa-4def-9f79-19d4cdd6d660,INTC,...
        """  # Replace this with actual data retrieval
        trade_history = parse_trade_data(raw_trade_data)
        
        if trade_history:
            display_trade_history(trade_history)
            export_to_csv(trade_history)
        else:
            logging.info("No trades to display or trade retrieval failed.")
    else:
        logging.error("Script halted due to failed login.")

# Run the script if executed directly
if __name__ == "__main__":
    main()
