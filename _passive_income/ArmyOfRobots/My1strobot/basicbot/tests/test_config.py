import unittest
from pathlib import Path
from unittest.mock import patch
from basicbot.config import load_config, ConfigModel, DataFetchingConfig


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_config_path = "temp_config.yaml"

    def tearDown(self):
        # Clean up temporary config file
        temp_path = Path(self.temp_config_path)
        if temp_path.exists():
            temp_path.unlink()

    @patch.dict("os.environ", {"DATA_FETCHING_FETCH_RETRIES": "5", "DATA_FETCHING_BACKOFF_FACTOR": "3.0"})
    def test_load_config_with_env_override(self):
        """
        Test loading configuration with environment variable overrides.
        """
        # Create a default config
        config = load_config(self.temp_config_path)

        # Validate overrides
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


if __name__ == "__main__":
    unittest.main()
