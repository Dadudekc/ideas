import pandas as pd
import logging
from typing import Any


class Backtester:
    def __init__(self, strategy, logger, api=None, symbol=None, timeframe=None, limit=None, portfolio=None, log_callback=None):
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
        - df: A DataFrame containing historical price data with a 'close' column.

        Returns:
        - DataFrame with added columns:
          - 'signal': The trading signal ('BUY', 'SELL', 'HOLD').
          - 'position': The position (1 for BUY, -1 for SELL, 0 for HOLD).
          - 'returns': Daily returns based on the 'close' prices.
          - 'strategy_returns': Strategy returns considering positions.
          - 'cumulative_returns': Cumulative returns of the strategy.
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
        return self.strategy.calculate_indicators(df)

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy.

        Parameters:
        - df: DataFrame with calculated indicators.

        Returns:
        - DataFrame with added 'signal' and 'position' columns.
        """
        self.logger.info("Generating signals.")
        df['signal'] = self.strategy.evaluate(df)
        df['position'] = df['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': None}).ffill().fillna(0)
        
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
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['returns'] * df['position'].shift(1)
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        return df
