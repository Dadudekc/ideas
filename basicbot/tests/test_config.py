import unittest
from pathlib import Path
from unittest.mock import patch
from basicbot.config import load_config, ConfigModel, DataFetchingConfig

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_config_path = "temp_config.yaml"

    def tearDown(self):
        temp_path = Path(self.temp_config_path)
        if temp_path.exists():
            temp_path.unlink()

    def test_load_config_with_env_override(self):
        """
        Test environment variable overrides.
        """
        with patch.dict("os.environ", {"DATA_FETCHING_FETCH_RETRIES": "5", "DATA_FETCHING_BACKOFF_FACTOR": "3.0"}):
            config = load_config(self.temp_config_path)
            self.assertEqual(config.data_fetching.fetch_retries, 5)
            self.assertEqual(config.data_fetching.backoff_factor, 3.0)

    def test_default_config_creation(self):
        """
        Test default configuration file creation.
        """
        load_config(self.temp_config_path)

        # Check if file exists
        temp_path = Path(self.temp_config_path)
        self.assertTrue(temp_path.exists())

        # Validate contents
        with open(self.temp_config_path, "r") as file:
            content = file.read()
        self.assertIn("symbol: TSLA", content)

    def test_recursive_env_override(self):
        """
        Test recursive environment variable overrides.
        """
        with patch.dict("os.environ", {"STRATEGY_RSI_OVERSOLD": "25"}):
            config = load_config(self.temp_config_path)
            self.assertEqual(config.strategy.rsi_oversold, 25.0)

    def test_env_override(self):
        """
        Test specific environment variable override.
        """
        with patch.dict("os.environ", {"STRATEGY_EMA_LENGTH": "20"}):
            config = load_config(self.temp_config_path)
            self.assertEqual(config.strategy.ema_length, 20)

    def test_load_config_with_invalid_file(self):
        """
        Test behavior when configuration file is invalid.
        """
        with open(self.temp_config_path, "w") as file:
            file.write("INVALID_CONTENT")

        with self.assertRaises(Exception):  # Replace with the specific exception
            load_config(self.temp_config_path)

    def test_load_config_with_missing_file(self):
        """
        Test behavior when configuration file is missing.
        """
        temp_path = Path(self.temp_config_path)
        if temp_path.exists():
            temp_path.unlink()

        config = load_config(self.temp_config_path)
        self.assertTrue(temp_path.exists())
        self.assertIn("symbol: TSLA", open(self.temp_config_path).read())

if __name__ == "__main__":
    unittest.main()
