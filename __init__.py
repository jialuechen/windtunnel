from .scenario_builder import ScenarioBuilder
from .regime_gen import RegimeGenerator
from .flow_gen import OrderFlowGenerator
from .lob_sim import LOBSimulator
from .strategy_executor import StrategyExecutor
from .impact_evaluator import ImpactEvaluator

class GenMarket:
    def __init__(self, config):
        self.config = config
        self.scenario = ScenarioBuilder(config).build()
        self.regime_gen = RegimeGenerator(self.scenario)
        self.flow_gen = OrderFlowGenerator()
        self.lob_sim = LOBSimulator(config.get('lob'))
        self.strategy_executor = StrategyExecutor(config.get('strategy'))
        self.impact_evaluator = ImpactEvaluator()

    def run(self):
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
