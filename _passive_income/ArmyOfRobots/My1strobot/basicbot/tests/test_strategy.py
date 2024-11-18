import unittest
import pandas as pd
import logging
from basicbot.strategy import Strategy
import sys
print("\n".join(sys.path))

class TestStrategy(unittest.TestCase):
    """
    Unit tests for the Strategy class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up common resources for all test cases.
        """
        cls.logger = logging.getLogger("TestStrategy")
        logging.basicConfig(level=logging.INFO)
        cls.config = {
            "strategy": {
                "ema_length": 8,
                "rsi_length": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30
            }
        }
        cls.strategy = Strategy(cls.config, cls.logger)

    def test_calculate_indicators(self):
        """
        Test the calculation of trading indicators.
        """
        # Provide at least 15 data points to exceed the maximum window size of 14
        data = {
            "close": [100, 102, 101, 104, 106, 105, 108, 107, 109, 110, 112, 113, 115, 117, 119]
        }
        df = pd.DataFrame(data)

        df_with_indicators = self.strategy.calculate_indicators(df)

        # Assertions
        self.assertIn("ema", df_with_indicators.columns)
        self.assertIn("rsi", df_with_indicators.columns)
        self.assertIn("macd", df_with_indicators.columns)
        self.assertIn("bollinger_hband", df_with_indicators.columns)
        self.assertIn("bollinger_lband", df_with_indicators.columns)
        self.assertFalse(df_with_indicators["ema"].isnull().all(), "EMA should not be all NaN")
        self.assertFalse(df_with_indicators["rsi"].isnull().all(), "RSI should not be all NaN")

    def test_evaluate_buy_signal(self):
        """
        Test evaluation logic for generating a BUY signal.
        """
        data = {
            "close": [100, 98, 96, 94, 92],
            "ema": [98, 97, 96, 95, 94],
            "rsi": [25, 28, 29, 27, 26],
            "macd": [0.5, 0.4, 0.3, 0.2, 0.1],
            "bollinger_hband": [105, 105, 105, 105, 105],
            "bollinger_lband": [90, 90, 90, 90, 90]
        }
        df = pd.DataFrame(data)

        signal = self.strategy.evaluate(df)

        # Assertions
        self.assertEqual(signal, 'BUY')

    def test_evaluate_sell_signal(self):
        """
        Test evaluation logic for generating a SELL signal.
        """
        data = {
            "close": [110, 112, 115, 118, 120],
            "ema": [105, 107, 110, 115, 118],
            "rsi": [75, 78, 80, 82, 85],
            "macd": [-0.2, -0.3, -0.4, -0.5, -0.6],
            "bollinger_hband": [125, 125, 125, 125, 125],
            "bollinger_lband": [95, 95, 95, 95, 95]
        }
        df = pd.DataFrame(data)

        signal = self.strategy.evaluate(df)

        # Assertions
        self.assertEqual(signal, 'SELL')


    def test_evaluate_hold_signal(self):
        """
        Test evaluation logic for generating a HOLD signal.
        """
        data = {
            "close": [100, 102, 104, 106, 108],
            "ema": [101, 103, 105, 107, 109],
            "rsi": [50, 52, 53, 54, 55],
            "macd": [0.1, 0.2, 0.3, 0.4, 0.5],
            "bollinger_hband": [115, 115, 115, 115, 115],
            "bollinger_lband": [85, 85, 85, 85, 85]
        }
        df = pd.DataFrame(data)

        signal = self.strategy.evaluate(df)

        # Assertions
        self.assertEqual(signal, 'HOLD')

if __name__ == "__main__":
    unittest.main()
