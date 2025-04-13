
from genmarket.regime_generator import RegimeGenerator
from genmarket.order_flow_generator import OrderFlowGenerator
from genmarket.lob_simulator import LOBSimulator  # Corrected module name
from genmarket.strategy_executor import StrategyExecutor
from genmarket.impact_evaluator import ImpactEvaluator

class GenMarket:
    def __init__(self, config):
        """
        Initialize the GenMarket simulation framework.

        Args:
            config (dict): Configuration dictionary for the simulation.
        """
        self.config = config or {}
        self.regime_gen = RegimeGenerator(self.config.get('regime', {}))
        self.flow_gen = OrderFlowGenerator()
        self.lob_sim = LOBSimulator(self.config.get('lob', {}))
        self.strategy_executor = StrategyExecutor(self.config.get('strategy', {}))
        self.impact_evaluator = ImpactEvaluator()

    def run(self):
        """
        Run the market simulation.

        Returns:
            dict: Impact metrics from the simulation.
        """
        print("Generating market regime...")
        regime = self.regime_gen.generate()

        print("Generating order flow...")
        orders = self.flow_gen.generate(regime)

        print("Simulating LOB dynamics...")
        lob_result = self.lob_sim.simulate(orders)

        print("Executing strategy...")
        strategy_result = self.strategy_executor.run(lob_result)

        print("Evaluating impact...")
        impact_metrics = self.impact_evaluator.evaluate(strategy_result)

        return impact_metrics