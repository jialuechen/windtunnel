import unittest
from unittest.mock import patch
from genmarket.nlp_config_generator import NLPConfigGenerator

class TestNLPConfigGenerator(unittest.TestCase):
    @patch("openai.Completion.create")
    def test_generate_config(self, mock_openai):
        # Mock OpenAI API response
        mock_openai.return_value = {
            "choices": [{"text": '{"market": "high-volatility", "liquidity": "low", "strategy": "VWAP", "target_volume": 1000, "duration": "1h"}'}]
        }

        # Initialize the generator
        generator = NLPConfigGenerator(api_key="test_api_key")

        # Example description
        description = "Simulate a high-volatility market with low liquidity. Use a VWAP strategy with a target volume of 1000 shares over a 1-hour period."

        # Generate configuration
        config = generator.generate_config(description)

        # Verify the generated configuration
        expected_config = '{"market": "high-volatility", "liquidity": "low", "strategy": "VWAP", "target_volume": 1000, "duration": "1h"}'
        self.assertEqual(config, expected_config)

if __name__ == "__main__":
    unittest.main()