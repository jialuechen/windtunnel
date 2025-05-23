import unittest
from windtunnel.impact_evaluator import ImpactEvaluator

class TestImpactEvaluator(unittest.TestCase):
    def test_calculate_slippage(self):
        slippage = ImpactEvaluator.calculate_slippage(expected_price=100.0, executed_price=102.0)
        self.assertAlmostEqual(slippage, 2.0, places=2)

    def test_calculate_impact_cost(self):
        executed_prices = [102, 103, 101]
        market_prices = [100, 101, 100]
        impact_cost = ImpactEvaluator.calculate_impact_cost(executed_prices, market_prices)
        self.assertAlmostEqual(impact_cost, 2.33, places=2)

    def test_calculate_pnl(self):
        pnl = ImpactEvaluator.calculate_pnl(entry_price=100.0, exit_price=105.0, position_size=10)
        self.assertEqual(pnl, 50.0)

if __name__ == "__main__":
    unittest.main()