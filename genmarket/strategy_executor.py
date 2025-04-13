import importlib
from genmarket.impact_evaluator import ImpactEvaluator

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

        # Example evaluation (replace with actual data as needed)
        expected_price = 100.0  # Example expected price
        executed_price = 102.0  # Example executed price
        slippage = ImpactEvaluator.calculate_slippage(expected_price, executed_price)

        return {
            "executions": executions,
            "strategy_param": self.config.get("param", "default"),
            "slippage": slippage
        }