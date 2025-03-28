class RegimeGenerator:
    def __init__(self, scenario_config):
        self.config = scenario_config

    def generate(self):
        return {
            "regime_vector": [0.1, 0.2, 0.3],
            "regime_type": self.config["regime_type"]
        }
