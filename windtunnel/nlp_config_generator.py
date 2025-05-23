import openai

class NLPConfigGenerator:
    def __init__(self, api_key):
        """
        Initialize the NLP Config Generator.

        Args:
            api_key (str): OpenAI API key for accessing GPT models.
        """
        openai.api_key = api_key

    def generate_config(self, description):
        """
        Generate a market simulation configuration from a natural language description.

        Args:
            description (str): Natural language description of the market scenario.

        Returns:
            dict: Generated configuration as a dictionary.
        """
        prompt = f"""
        Convert the following market scenario description into a JSON configuration:
        Description: {description}
        Configuration:
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        config_text = response['choices'][0]['text'].strip()
        return config_text