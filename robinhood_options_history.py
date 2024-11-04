import logging
import robin_stocks.robinhood as rh
import csv
import os
import requests
import concurrent.futures
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Optional, List, Dict
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Load configuration from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# HTTP Session for retries and timeouts
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)

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

def fetch_url_with_retries(url: str) -> Optional[Dict]:
    """Fetches JSON data from a URL with retries and timeout handling."""
    if not url:
        logging.warning("Received an empty URL, skipping fetch.")
        return None
    try:
        logging.info(f"Fetching data from URL: {url}")
        response = http.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logging.error(f"Request to {url} timed out.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from {url}: {e}")
    return None

def enrich_trade_history_parallel(trade_history: List[Dict]) -> List[Dict]:
    """Fetches additional data from the option instrument URL in parallel."""
    logging.info("Enriching trade data with additional details in parallel...")
    enriched_history = []

    # Skip any URL fetching and directly add trade details as is
    for trade in trade_history:
        # Ensure minimal fields are populated without attempting any external data fetch
        enriched_trade = validate_trade_data(trade)
        enriched_history.append(enriched_trade)
    
    logging.info("Trade data enrichment completed without URL fetching.")
    return enriched_history



def validate_trade_data(trade: Dict) -> Dict:
    """Ensures required fields have default values if missing."""
    required_fields = {
        "option_type": "N/A",
        "expiration_date": "N/A",
        "strike_price": "N/A",
        "multiplier": 100,
        "chain_symbol": "N/A",
    }
    for field, default in required_fields.items():
        trade[field] = trade.get(field, default)
    return trade


def filter_recent_trades(trade_history: List[Dict], days: int = 90) -> List[Dict]:
    """Filters trade history to include only trades within the last specified days."""
    recent_trades = []
    cutoff_date = datetime.now() - timedelta(days=days)
    for trade in trade_history:
        updated_at = trade.get("updated_at")
        if updated_at:
            trade_date = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%S.%fZ")
            if trade_date >= cutoff_date:
                recent_trades.append(trade)
    logging.info(f"Filtered {len(recent_trades)} trades within the last {days} days.")
    return recent_trades

def export_to_csv(trade_history: List[Dict], filename: str = "enriched_options_trade_history.csv") -> None:
    """Exports the trade history to a CSV file."""
    if not trade_history:
        logging.warning("No trade history to export.")
        return
    
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=trade_history[0].keys())
            writer.writeheader()
            writer.writerows(trade_history)
        logging.info(f"Trade history successfully exported to {filename} with {len(trade_history)} records.")
    except Exception as e:
        logging.error(f"Failed to export trade history to CSV: {e}")

def analyze_csv_data(filename: str = "enriched_options_trade_history.csv") -> None:
    """Performs in-depth analysis on the exported CSV data and generates visualizations."""
    df = pd.read_csv(filename)
    
    # Display data types and missing values
    print("Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())

    # Filter and group data for detailed analysis
    tsla_df = df[df['chain_symbol'] == 'TSLA']
    account_groups = df.groupby('account_number')
    option_groups = df.groupby('option_id')

    # Average price per account
    average_prices = account_groups['average_price'].mean()
    print("\nAverage prices per account:")
    print(average_prices)

    # Pending buy/sell quantities per account
    total_pending_quantity = account_groups[['pending_buy_quantity', 'pending_sell_quantity']].sum()
    print("\nTotal pending quantities per account:")
    print(total_pending_quantity)

    # Debit/Credit summary
    debit_credit_summary = df.groupby('clearing_direction')['average_price'].sum()
    print("\nDebit/Credit summary:")
    print(debit_credit_summary)

    # Date-based analysis for expiration
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    df['DTE'] = (df['expiration_date'] - datetime.now()).dt.days

    active_options = df[df['DTE'] > 0]
    expired_options = df[df['DTE'] <= 0]
    print("\nActive options:")
    print(active_options)
    print("\nExpired options:")
    print(expired_options)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(df['DTE'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Days to Expiration')
    plt.xlabel('Days to Expiration')
    plt.ylabel('Number of Options')
    plt.show()

    plt.figure(figsize=(8, 6))
    debit_credit_summary.plot(kind='bar', color=['blue', 'orange'], edgecolor='black')
    plt.title('Debit and Credit Analysis')
    plt.ylabel('Total Amount')
    plt.show()

    total_pending_quantity.plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'red'])
    plt.title('Pending Buy vs. Sell Quantities per Account')
    plt.ylabel('Quantity')
    plt.show()

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Retrieve and export Robinhood options trade history.")
    parser.add_argument("--days", type=int, default=30, help="Number of days for recent trades filtering.")
    parser.add_argument("--export_file", type=str, default="enriched_options_trade_history.csv", help="CSV export filename.")
    return parser.parse_args()

def main():
    """Main function to handle login, retrieval, display, enrichment, and export of trade history."""
    args = parse_args()
    username = os.getenv("ROBINHOOD_USERNAME")
    password = os.getenv("ROBINHOOD_PASSWORD")

    if not username or not password:
        logging.error("Username or password not set. Please configure them in the .env file.")
        return

    if login(username, password):
        trade_history = get_all_options_trade_history()
        if trade_history:
            recent_trades = filter_recent_trades(trade_history, days=args.days)
            enriched_trade_history = enrich_trade_history_parallel(recent_trades)
            if enriched_trade_history:  # Ensure there is data to export
                export_to_csv(enriched_trade_history, filename=args.export_file)
                analyze_csv_data(filename=args.export_file)  # Perform analysis on exported data
            else:
                logging.warning("No enriched trade history to export.")
        else:
            logging.info("No trades to display or trade retrieval failed.")
    else:
        logging.error("Script halted due to failed login.")

if __name__ == "__main__":
    main()
