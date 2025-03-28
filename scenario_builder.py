class ScenarioBuilder:
    def __init__(self, config):
        self.config = config

    def build(self):
        return {
            "regime_type": self.config.get("regime", "normal"),
            "volatility": self.config.get("volatility", 0.2),
            "liquidity": self.config.get("liquidity", "medium"),
        }
