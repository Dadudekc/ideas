import unittest
from pydantic import ValidationError
from basicbot.strategy_config import StrategyConfig


class TestStrategyConfig(unittest.TestCase):
    def test_default_values(self):
        """
        Test that default values are correctly applied.
        """
        config = StrategyConfig()
        self.assertEqual(config.ema_length, 8)
        self.assertEqual(config.rsi_length, 14)
        self.assertEqual(config.rsi_overbought, 70.0)
        self.assertEqual(config.rsi_oversold, 30.0)
        self.assertEqual(config.macd_fast_window, 12)
        self.assertEqual(config.macd_slow_window, 26)
        self.assertEqual(config.macd_signal_window, 9)
        self.assertEqual(config.adx_window, 14)
        self.assertEqual(config.vwap_window, 14)
        self.assertEqual(config.bollinger_window, 20)
        self.assertEqual(config.atr_window, 14)
        self.assertEqual(config.risk_percent, 0.5)
        self.assertEqual(config.profit_target, 15.0)
        self.assertEqual(config.stop_loss_multiplier, 2.0)

    def test_validation_success(self):
        """
        Test that valid configurations pass validation.
        """
        config = StrategyConfig(
            ema_length=10,
            rsi_length=20,
            rsi_overbought=80.0,
            rsi_oversold=20.0,
            macd_fast_window=15,
            macd_slow_window=30,
            macd_signal_window=10,
            adx_window=20,
            vwap_window=10,
            bollinger_window=25,
            atr_window=10,
            risk_percent=1.0,
            profit_target=20.0,
            stop_loss_multiplier=1.5,
        )
        self.assertEqual(config.ema_length, 10)
        self.assertEqual(config.rsi_length, 20)
        self.assertEqual(config.rsi_overbought, 80.0)
        self.assertEqual(config.rsi_oversold, 20.0)

    def test_validation_failure(self):
        """
        Test that invalid configurations raise ValidationError.
        """
        with self.assertRaises(ValidationError):
            StrategyConfig(ema_length=0)  # Should fail due to `ge=1`

        with self.assertRaises(ValidationError):
            StrategyConfig(rsi_overbought=101.0)  # Should fail due to `le=100`

        with self.assertRaises(ValidationError):
            StrategyConfig(risk_percent=200.0)  # Should fail due to `le=100`

        with self.assertRaises(ValidationError):
            StrategyConfig(stop_loss_multiplier=0.5)  # Should fail due to `ge=1`

    def test_serialization(self):
        """
        Test that the configuration can be serialized to a dictionary and back.
        """
        config = StrategyConfig()
        config_dict = config.model_dump()  # Use model_dump instead of dict
        new_config = StrategyConfig(**config_dict)
        self.assertEqual(config, new_config)

    def test_partial_config(self):
        """
        Test creating a config with partial fields provided.
        """
        config = StrategyConfig(ema_length=10, profit_target=25.0)
        self.assertEqual(config.ema_length, 10)
        self.assertEqual(config.profit_target, 25.0)
        # Ensure defaults are applied to other fields
        self.assertEqual(config.rsi_length, 14)
        self.assertEqual(config.stop_loss_multiplier, 2.0)


if __name__ == "__main__":
    unittest.main()
