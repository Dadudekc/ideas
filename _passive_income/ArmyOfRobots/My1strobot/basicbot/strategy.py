import pandas as pd
import ta
import logging
from typing import Optional, Dict, Any, List

class Strategy:
    """
    Strategy class encapsulates the parameters and logic required for trading strategies.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Strategy class with provided parameters.
        """
        strategy_params = config.get("strategy", {})
        self.ema_length = strategy_params.get("ema_length", 8)
        self.rsi_length = strategy_params.get("rsi_length", 14)
        self.rsi_overbought = strategy_params.get("rsi_overbought", 70)
        self.rsi_oversold = strategy_params.get("rsi_oversold", 30)
        self.logger = logger

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading indicators on the provided DataFrame.
        """
        self.logger.info("Calculating indicators.")

        if len(df) < max(self.ema_length, self.rsi_length):
            self.logger.warning("Not enough data to calculate indicators. Returning DataFrame with NaN indicators.")
            df['ema'] = pd.Series([float('nan')] * len(df))
            df['rsi'] = pd.Series([float('nan')] * len(df))
            df['macd'] = pd.Series([float('nan')] * len(df))
            df['bollinger_hband'] = pd.Series([float('nan')] * len(df))
            df['bollinger_lband'] = pd.Series([float('nan')] * len(df))
            return df

        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=self.ema_length).ema_indicator()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_length).rsi()
        # Additional indicators can be added here
        macd_indicator = ta.trend.MACD(close=df['close'])
        df['macd'] = macd_indicator.macd()
        bollinger = ta.volatility.BollingerBands(close=df['close'])
        df['bollinger_hband'] = bollinger.bollinger_hband()
        df['bollinger_lband'] = bollinger.bollinger_lband()
        return df

    def evaluate(self, df: pd.DataFrame) -> str:
        """
        Evaluate the latest row in the DataFrame and generate a single trading signal.
        """
        self.logger.info("Evaluating strategy based on indicators.")
        if df.empty or 'rsi' not in df.columns or df['rsi'].isnull().all():
            self.logger.warning("DataFrame is empty or indicators are not calculated. Returning HOLD signal.")
            return 'HOLD'

        # Evaluate only the latest row
        row = df.iloc[-1]
        rsi_value = row['rsi']
        ema_value = row['ema']
        close_price = row['close']

        if pd.isna(rsi_value) or pd.isna(ema_value) or pd.isna(close_price):
            signal = 'HOLD'
        elif rsi_value < self.rsi_oversold and close_price < ema_value:
            signal = 'BUY'
            self.logger.info(f"BUY signal generated: RSI oversold ({rsi_value}) and price below EMA ({close_price} < {ema_value}).")
        elif rsi_value > self.rsi_overbought and close_price > ema_value:
            signal = 'SELL'
            self.logger.info(f"SELL signal generated: RSI overbought ({rsi_value}) and price above EMA ({close_price} > {ema_value}).")
        else:
            signal = 'HOLD'
            self.logger.info(f"HOLD signal generated: RSI is neutral ({rsi_value}).")

        self.logger.info(f"Generated signal: {signal}.")
        return signal
