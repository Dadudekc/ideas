import pandas as pd
import logging
from typing import Any

class Backtester:
    def __init__(
        self,
        strategy,
        logger: logging.Logger,
        api: Any = None,
        symbol: str = None,
        timeframe: str = None,
        limit: int = None,
        portfolio: Any = None,
        log_callback: Any = None
    ):
        """
        Initialize the Backtester with additional parameters for flexibility.

        Parameters:
        - strategy: Strategy instance.
        - logger: Logger instance.
        - api (optional): API object for fetching data (default None).
        - symbol (optional): Symbol for backtesting (default None).
        - timeframe (optional): Timeframe for backtesting (default None).
        - limit (optional): Data limit for backtesting (default None).
        - portfolio (optional): Portfolio details (default None).
        - log_callback (optional): Callback for custom logging (default None).
        """
        self.strategy = strategy
        self.logger = logger
        self.api = api
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.portfolio = portfolio
        self.log_callback = log_callback

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the backtest and return a DataFrame with signals and portfolio performance.

        Parameters:
        - df: A DataFrame containing historical price data with 'high', 'low', and 'close' columns.

        Returns:
        - DataFrame with added columns:
          - 'signal': The trading signal ('BUY', 'SELL', 'HOLD').
          - 'position': The position (1 for BUY, -1 for SELL, 0 for HOLD).
          - 'returns': Daily returns based on the 'close' prices.
          - 'strategy_returns': Strategy returns considering positions.
          - 'cumulative_returns': Cumulative returns of the strategy.
          - 'atr': Average True Range indicator.
        """
        self.logger.info("Starting backtest.")
        df = self._calculate_indicators(df)
        df = self._generate_signals(df)
        df = self._calculate_returns(df)
        self.logger.info("Backtest completed.")
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the required indicators for the backtest.

        Parameters:
        - df: Input DataFrame with historical price data.

        Returns:
        - DataFrame with calculated indicators.
        """
        self.logger.info("Calculating indicators.")

        # Ensure all necessary columns are present
        required_cols = ["high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            self.logger.error(f"DataFrame must contain columns: {required_cols}")
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Calculate True Range (TR)
        df['tr'] = df[['high', 'low', 'close']].apply(
            lambda row: max(
                row['high'] - row['low'],
                abs(row['high'] - row['close']),
                abs(row['low'] - row['close'])
            ),
            axis=1
        )
        # Calculate Average True Range (ATR) with a 14-day window
        df['atr'] = df['tr'].rolling(window=14, min_periods=1).mean()

        # Delegate additional indicator calculations to the strategy
        df = self.strategy.calculate_indicators(df)

        return df

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy and determine positions statefully.

        Parameters:
        - df: DataFrame with calculated indicators.

        Returns:
        - DataFrame with updated 'signal' and 'position' columns.
        """
        self.logger.info("Generating signals.")
        df['signal'] = self.strategy.evaluate(df)

        positions = []
        current_position = 0  # 1 for LONG, -1 for SHORT, 0 for NO POSITION

        for signal in df['signal']:
            if signal == 'BUY':
                if current_position == 0:
                    current_position = 1  # Enter LONG
                    self.logger.debug("BUY signal received. Entering LONG position.")
                elif current_position == -1:
                    current_position = 1  # Reverse to LONG from SHORT
                    self.logger.debug("BUY signal received. Reversing from SHORT to LONG position.")
                # If already in LONG, maintain position
            elif signal == 'SELL':
                if current_position == 1:
                    current_position = 0  # Exit LONG
                    self.logger.debug("SELL signal received. Exiting LONG position.")
                elif current_position == 0:
                    current_position = -1  # Enter SHORT
                    self.logger.debug("SELL signal received. Entering SHORT position.")
                elif current_position == -1:
                    current_position = 0  # Exit SHORT
                    self.logger.debug("SELL signal received. Exiting SHORT position.")
            elif signal == 'HOLD':
                self.logger.debug("HOLD signal received. Maintaining current position.")
                pass  # Maintain current position
            else:
                self.logger.warning(f"Unknown signal '{signal}'. Maintaining current position.")

            positions.append(current_position)

        df['position'] = positions

        # Debug output to verify 'signal' and 'position'
        print(df[['signal', 'position']])

        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns and cumulative strategy performance.

        Parameters:
        - df: DataFrame with signals and positions.

        Returns:
        - DataFrame with added 'returns', 'strategy_returns', and 'cumulative_returns' columns.
        """
        self.logger.info("Calculating returns and performance.")
        df['returns'] = df['close'].pct_change().fillna(0)  # Avoid NaN in returns
        df['strategy_returns'] = (df['returns'] * df['position'].shift(1)).fillna(0)
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod().fillna(1)  # Start cumulative at 1
        return df
