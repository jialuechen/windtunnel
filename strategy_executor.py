class StrategyExecutor:
    def __init__(self, config):
        self.config = config or {}

    def run(self, lob_result):
        return {
            "executions": lob_result,
            "strategy_param": self.config.get("param", "default")
        }
