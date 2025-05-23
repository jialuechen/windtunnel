from windtunnel.nlp_config_generator import NLPConfigGenerator

# Initialize the NLP Config Generator with your OpenAI API key
api_key = "your_openai_api_key"  # Replace with your actual API key
generator = NLPConfigGenerator(api_key=api_key)

# Example natural language description
description = """
Simulate a high-volatility market with low liquidity. Use a VWAP strategy with a target volume of 1000 shares over a 1-hour period.
"""

# Generate configuration
config = generator.generate_config(description)

# Print the generated configuration
print("Generated Configuration:")
print(config)

# Optionally, save the configuration to a JSON file
with open("market_simulation_config.json", "w") as f:
    f.write(config)