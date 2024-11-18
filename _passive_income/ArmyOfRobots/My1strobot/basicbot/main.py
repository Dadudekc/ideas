import logging
import sys
from pathlib import Path

# Add project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from basicbot.logger import setup_logger
from basicbot.config import load_config
from basicbot.data_fetcher import DataFetchUtils
from basicbot.strategy import Strategy
from basicbot.trader import Trader
from basicbot.trading_api import TradingAPI
from basicbot.notifier import Notifier

def main():
    # Step 1: Load Configuration
    config = load_config("1config.yaml")
    
    # Step 2: Setup Logger
    logger = setup_logger("BasicBot", config.logging)
    
    # Step 3: Initialize Components
    logger.info("Initializing components...")
    data_fetcher = DataFetchUtils(api_key=config.data.api_key, logger=logger)
    strategy = Strategy(logger=logger)
    trading_api = TradingAPI(api_key=config["trading"]["api_key"], logger=logger)
    trader = Trader(api=trading_api, logger=logger)
    notifier = Notifier(config=config["notifier"], logger=logger)
    
    try:
        # Step 4: Fetch Market Data
        logger.info("Fetching market data...")
        symbol = config["trading"]["symbol"]
        start_date = config["data"]["start_date"]
        end_date = config["data"]["end_date"]
        interval = config["data"]["interval"]
        
        market_data = data_fetcher.fetch_stock_data_async(symbol, start_date, end_date, interval)
        logger.info(f"Fetched data for {symbol} from {start_date} to {end_date}.")
        
        # Step 5: Generate Trading Signals
        logger.info("Generating trading signals...")
        signals = strategy.generate_signals(market_data)
        logger.info(f"Generated signals: {signals}")
        
        # Step 6: Execute Trades
        logger.info("Executing trades...")
        current_price = market_data["close"].iloc[-1]  # Assuming 'close' column exists
        qty = config["trading"]["quantity"]
        trader.execute_signals(signals, symbol, current_price, qty)
        
        # Step 7: Notify User
        logger.info("Sending notifications...")
        notifier.send_notification("Trading activity completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        notifier.send_notification(f"An error occurred: {e}")
    
    logger.info("BasicBot execution completed.")

if __name__ == "__main__":
    main()
