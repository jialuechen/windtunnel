import importlib

class StrategyExecutor:
    def __init__(self, config):
        self.config = config or {}

    def run(self, lob_result):
        executions = lob_result
        strategy_type = self.config.get("type")
        if strategy_type in ("vwap", "twap", "dynamic_stop_loss"):
            # Dynamically import the strategy module
            module = importlib.import_module(f"genmarket.strategy_plugins.{strategy_type}")
            StrategyClass = getattr(module, f"{strategy_type.replace('_', '').capitalize()}Strategy")
            strategy = StrategyClass(**self.config.get("params", {}))
            executions = strategy.generate_orders(lob_result)

        return {
            "executions": executions,
            "strategy_param": self.config.get("param", "default")
        }