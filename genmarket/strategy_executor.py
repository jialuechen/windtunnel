import importlib

class StrategyExecutor:
    def __init__(self, config):
        self.config = config or {}

    def run(self, lob_result):
        executions = lob_result
        if self.config.get("type") in ("vwap", "twap"):
            module = importlib.import_module(f"genmarket.strategy_plugins.{self.config['type']}")
            StrategyClass = getattr(module, f"{self.config['type'].upper()}Strategy")
            strategy = StrategyClass(**self.config.get("params", {}))
            executions = strategy.generate_orders(lob_result)

        return {
            "executions": executions,
            "strategy_param": self.config.get("param", "default")
        }