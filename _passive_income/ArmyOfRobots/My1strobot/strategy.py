# strategy.py

import pandas as pd
import ta
import logging
from typing import List

class Strategy:
    """
    Strategy class encapsulates the parameters and logic required for trading strategies.

    Attributes:
        vwap_session (str): Session type for VWAP calculation.
        ema_length (int): Length of EMA for calculation.
        rsi_length (int): Length for RSI calculation.
        rsi_overbought (int): RSI overbought threshold.
        rsi_oversold (int): RSI oversold threshold.
        macd_fast (int): Fast EMA length for MACD calculation.
        macd_slow (int): Slow EMA length for MACD calculation.
        macd_signal (int): Signal line EMA length for MACD.
        adx_length (int): Length for ADX calculation.
        volume_threshold_length (int): Length for volume threshold calculation.
        atr_length (int): Length for ATR calculation.
        risk_percent (float): Percentage of portfolio to risk per trade.
        profit_target_percent (float): Profit target as a percentage.
        stop_multiplier (float): Stop-loss multiplier.
        trail_multiplier (float): Trailing stop multiplier.
        timeframe (str): Timeframe for strategy evaluation.
        limit (int): Limit for historical data fetching.
        required_length (int): Maximum length of data required for calculations.
        logger (Logger): Logger instance for the strategy class.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Strategy class with provided parameters.

        :param kwargs: Dictionary of parameters to initialize the strategy.
        """
        # Initialize strategy parameters with validation
        self.vwap_session = self._validate_string_param(
            kwargs.get("vwap_session", "RTH"), "vwap_session"
        )
        self.ema_length = self._validate_numeric_param(
            kwargs.get("ema_length", 8), "ema_length", min_value=1
        )
        self.rsi_length = self._validate_numeric_param(
            kwargs.get("rsi_length", 14), "rsi_length", min_value=1
        )
        self.rsi_overbought = self._validate_numeric_param(
            kwargs.get("rsi_overbought", 70), "rsi_overbought", min_value=0, max_value=100
        )
        self.rsi_oversold = self._validate_numeric_param(
            kwargs.get("rsi_oversold", 30), "rsi_oversold", min_value=0, max_value=100
        )
        self.macd_fast = self._validate_numeric_param(
            kwargs.get("macd_fast", 12), "macd_fast", min_value=1
        )
        self.macd_slow = self._validate_numeric_param(
            kwargs.get("macd_slow", 26), "macd_slow", min_value=1
        )
        self.macd_signal = self._validate_numeric_param(
            kwargs.get("macd_signal", 9), "macd_signal", min_value=1
        )
        self.adx_length = self._validate_numeric_param(
            kwargs.get("adx_length", 14), "adx_length", min_value=1
        )
        self.volume_threshold_length = self._validate_numeric_param(
            kwargs.get("volume_threshold_length", 20), "volume_threshold_length", min_value=1
        )
        self.atr_length = self._validate_numeric_param(
            kwargs.get("atr_length", 14), "atr_length", min_value=1
        )
        self.risk_percent = self._validate_numeric_param(
            kwargs.get("risk_percent", 1.0), "risk_percent", min_value=0, max_value=100
        )
        self.profit_target_percent = self._validate_numeric_param(
            kwargs.get("profit_target_percent", 2.0), "profit_target_percent", min_value=0
        )
        self.stop_multiplier = self._validate_numeric_param(
            kwargs.get("stop_multiplier", 1.5), "stop_multiplier", min_value=0
        )
        self.trail_multiplier = self._validate_numeric_param(
            kwargs.get("trail_multiplier", 1.0), "trail_multiplier", min_value=0
        )
        self.timeframe = self._validate_string_param(
            kwargs.get("timeframe", "1D"), "timeframe"
        )
        self.limit = self._validate_numeric_param(
            kwargs.get("limit", 1000), "limit", min_value=1
        )

        # Calculate the required length for indicators
        self.required_length = max(
            self.ema_length,
            self.rsi_length,
            self.macd_slow,
            self.adx_length,
            self.volume_threshold_length,
            self.atr_length,
        ) + 1  # +1 for accurate calculations


        # Initialize logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger for the strategy.

        :return: Configured logger instance.
        """
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return logger

    def _validate_numeric_param(self, value, name, min_value=None, max_value=None):
        """
        Validate a numeric parameter.

        :param value: Parameter value to validate.
        :param name: Name of the parameter.
        :param min_value: Minimum allowed value (inclusive).
        :param max_value: Maximum allowed value (inclusive).
        :return: Validated parameter value.
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Parameter '{name}' must be a numeric value. Got: {value}")
        if min_value is not None and value < min_value:
            raise ValueError(
                f"Parameter '{name}' must be at least {min_value}. Got: {value}"
            )
        if max_value is not None and value > max_value:
            raise ValueError(
                f"Parameter '{name}' must not exceed {max_value}. Got: {value}"
            )
        return value

    def _validate_string_param(self, value, name):
        """
        Validate a string parameter.

        :param value: Parameter value to validate.
        :param name: Name of the parameter.
        :return: Validated parameter value.
        """
        if not isinstance(value, str):
            raise ValueError(f"Parameter '{name}' must be a string. Got: {value}")
        return value

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Starting indicator calculations.")

        # Validate and clean the DataFrame
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing columns for indicator calculations: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        # Drop rows with NaN in essential columns for specific indicators, not globally
        df_clean = df.copy()
        for col in required_columns:
            if df[col].isnull().any():
                self.logger.warning(f"Column '{col}' contains NaN values. Dropping only where necessary.")
                df_clean[col].fillna(method='bfill', inplace=True)

        # Ensure sufficient data for indicators
        if len(df_clean) < self.required_length:
            self.logger.warning(
                f"Not enough data to calculate indicators. Required: {self.required_length}, Available: {len(df_clean)}"
            )
            return df_clean

        try:
            self._calculate_vwap(df_clean)
            self._calculate_ema(df_clean)
            self._calculate_rsi(df_clean)
            self._calculate_macd(df_clean)
            self._calculate_adx(df_clean)
            self.logger.debug("All indicators calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            raise

        return df_clean

    def _calculate_vwap(self, df: pd.DataFrame):
        """Calculate Volume Weighted Average Price (VWAP)."""
        self.logger.debug("Calculating VWAP.")
        try:
            vwap_indicator = ta.volume.VolumeWeightedAveragePrice(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14  # VWAP typically uses a fixed window
            )
            df['vwap'] = vwap_indicator.volume_weighted_average_price()
            self.logger.debug("VWAP calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            raise

    def _calculate_ema(self, df: pd.DataFrame):
        """Calculate Exponential Moving Average (EMA)."""
        self.logger.debug("Calculating EMA.")
        try:
            ema_indicator = ta.trend.EMAIndicator(close=df['close'], window=self.ema_length)
            df['ema'] = ema_indicator.ema_indicator()
            self.logger.debug("EMA calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            raise

    def _calculate_rsi(self, df: pd.DataFrame):
        """Calculate Relative Strength Index (RSI)."""
        self.logger.debug("Calculating RSI.")
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_length)
            df['rsi'] = rsi_indicator.rsi()
            self.logger.debug("RSI calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            raise

    def _calculate_macd(self, df: pd.DataFrame):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        self.logger.debug("Calculating MACD.")
        try:
            macd = ta.trend.MACD(
                close=df['close'],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal
            )
            df['macd_line'] = macd.macd()
            df['signal_line'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            self.logger.debug("MACD calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            raise

    def _calculate_adx(self, df: pd.DataFrame):
        """
        Calculate the ADX indicator and add it to the DataFrame.

        :param df: DataFrame containing the historical data with 'high', 'low', 'close' columns.
        """
        self.logger.debug("Calculating ADX.")
        try:
            # Ensure required columns are present
            required_columns = ['high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' for ADX calculation.")

            # Strictly ensure there is enough data before calculation
            if len(df) < self.adx_length:
                self.logger.warning(
                    f"Not enough data to calculate ADX. Required: {self.adx_length}, Available: {len(df)}"
                )
                # Fill 'adx' column with NaN values
                df['adx'] = pd.Series([float('nan')] * len(df), index=df.index)
                return

            # Proceed with ADX calculation if data length is sufficient
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.adx_length
            )
            df['adx'] = adx_indicator.adx()
            self.logger.debug("ADX calculated successfully.")

        except IndexError as e:
            self.logger.error(f"IndexError during ADX calculation: {e}")
            # Ensure an 'adx' column exists to prevent further errors
            df['adx'] = pd.Series([float('nan')] * len(df), index=df.index)

        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}", exc_info=True)
            df['adx'] = pd.Series([float('nan')] * len(df), index=df.index)


    def _calculate_volume_threshold(self, df: pd.DataFrame):
        """Calculate Volume Threshold based on rolling mean."""
        self.logger.debug("Calculating Volume Threshold.")
        try:
            df['volume_threshold'] = df['volume'].rolling(window=self.volume_threshold_length).mean()
            self.logger.debug("Volume Threshold calculated successfully.")
        except Exception as e:
            self.logger.error(f"Error calculating Volume Threshold: {e}")
            raise

    def _calculate_atr(self, df: pd.DataFrame):
        """Calculate Average True Range (ATR)."""
        self.logger.debug("Calculating ATR.")
        try:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=self.atr_length
            )
            df['atr'] = atr_indicator.average_true_range()
            self.logger.debug(f"ATR calculated successfully. Sample values:\n{df['atr'].tail()}")

            # Validate ATR
            if df['atr'].isnull().all():
                self.logger.error("ATR column contains only NaN values.")
                raise ValueError("ATR calculation failed or contains invalid data.")
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            raise

    def evaluate(self, df: pd.DataFrame) -> List[str]:
        """
        Evaluate the latest indicators and generate trading signals.

        Parameters:
            df (pd.DataFrame): The DataFrame slice up to the current point in time.

        Returns:
            List[str]: A list of trading signals, e.g., ['BUY'] or ['SELL'].
        """
        self.logger.debug("Evaluating strategy based on indicators.")
        signals = []

        if df.empty:
            self.logger.debug("DataFrame slice is empty. No action taken.")
            return signals

        # Ensure required indicators are present in the DataFrame
        required_indicators = ['rsi', 'adx']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        if missing_indicators:
            self.logger.error(f"Missing indicators for evaluation: {missing_indicators}")
            return signals

        # Get the latest indicator values
        latest = df.iloc[-1]

        try:
            rsi_value = latest.get('rsi', None)
            adx_value = latest.get('adx', None)

            # Log indicator values for debugging
            self.logger.debug(f"Latest RSI: {rsi_value}, Latest ADX: {adx_value}")

            # Validate indicators before performing comparisons
            if pd.isnull(rsi_value) or pd.isnull(adx_value):
                self.logger.warning("Invalid RSI or ADX value detected. Skipping evaluation.")
                return signals

            # Example Strategy Logic:
            # Buy signal: RSI below oversold and ADX above a threshold (e.g., 25)
            if (rsi_value < self.rsi_oversold) and (adx_value > 25):
                signals.append('BUY')
                self.logger.debug("Buy signal generated.")

            # Sell signal: RSI above overbought and ADX above a threshold (e.g., 25)
            if (rsi_value > self.rsi_overbought) and (adx_value > 25):
                signals.append('SELL')
                self.logger.debug("Sell signal generated.")

        except Exception as e:
            self.logger.error(f"Error during signal evaluation: {e}")

        return signals

def main():
    # Extended dataset with at least 27 rows
    data = {
        'high': [100, 102, 101, 104, 106, 108, 110, 112, 115, 117, 120, 123, 125, 127, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 158, 160],
        'low': [98, 99, 97, 100, 105, 107, 109, 111, 113, 115, 118, 121, 123, 125, 128, 130, 133, 136, 138, 140, 143, 146, 148, 150, 153, 156, 158],
        'close': [99, 101, 100, 103, 105, 107, 110, 111, 114, 116, 119, 122, 124, 126, 129, 131, 134, 137, 139, 141, 144, 147, 149, 151, 154, 157, 159],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600],
    }
    df = pd.DataFrame(data)

    # Derive 'open' column
    df['open'] = df['close'].shift(1)
    df['open'] = df['open'].bfill()

    print("DataFrame with 'open' derived:")
    print(df)

    # Initialize the strategy with required parameters
    strategy = Strategy(
        vwap_session="RTH",
        ema_length=8,
        rsi_length=14,
        rsi_overbought=70,
        rsi_oversold=30,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        adx_length=14,
        volume_threshold_length=20,
        atr_length=14,
        risk_percent=1.0,
        profit_target_percent=2.0,
        stop_multiplier=1.5,
        trail_multiplier=1.0,
        timeframe="1D",
        limit=1000
    )

    # Run calculate_indicators
    try:
        df_with_indicators = strategy.calculate_indicators(df)
        print("\nDataFrame with indicators:")
        print(df_with_indicators.tail())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
