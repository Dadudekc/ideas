import logging
import sys
from pathlib import Path

# Add parent directory to sys.path for importing modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import modules
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
    logger = setup_logger("BasicBot", config.get("logging", {}))
    
    # Step 3: Initialize Components
    logger.info("Initializing components...")
    try:
        data_fetcher = DataFetchUtils(logger=logger, config=config.get("data_fetching", {}))
        strategy = Strategy(config=config.get("strategy", {}), logger=logger)
        trading_api = TradingAPI(
            api_key=config.notification.get("api_key", ""),
            api_secret=config.notification.get("api_secret", ""),
            base_url=config.notification.get("base_url", ""),
            logger=logger
        )
        trader = Trader(api=trading_api, logger=logger)
        notifier = Notifier(config=config.get("notification", {}), logger=logger)
    except KeyError as e:
        logger.error(f"Configuration missing required key: {e}")
        raise

    try:
        # Step 4: Fetch Market Data
        logger.info("Fetching market data...")
        symbol = config.get("symbol", "AAPL")  # Default to AAPL if not provided
        start_date = config["data_fetching"].get("start_date", "2020-01-01")
        end_date = config["data_fetching"].get("end_date", "2023-01-01")
        interval = config["data_fetching"].get("interval", "1d")
        
        market_data = data_fetcher.fetch_stock_data(symbol, start_date, end_date, interval)
        if market_data.empty:
            logger.warning(f"No market data fetched for {symbol}. Exiting.")
            return
        
        logger.info(f"Fetched data for {symbol} from {start_date} to {end_date}.")
        
        # Step 5: Generate Trading Signals
        logger.info("Generating trading signals...")
        market_data = strategy.calculate_indicators(market_data)
        signals = strategy.evaluate(market_data)
        if not signals:
            logger.info("No actionable trading signals generated.")
            return
        
        logger.info(f"Generated signals: {signals}")
        
        # Step 6: Execute Trades
        logger.info("Executing trades...")
        current_price = market_data["close"].iloc[-1]  # Assuming 'close' column exists
        qty = config.get("limit", 1)  # Default to 1 if not provided
        trader.execute_signals(signals, symbol, current_price, qty)
        
        # Step 7: Notify User
        logger.info("Sending notifications...")
        notifier.send_notification("Trading activity completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        notifier.send_notification(f"An error occurred: {e}")
    finally:
        logger.info("BasicBot execution completed.")

if __name__ == "__main__":
    main()
