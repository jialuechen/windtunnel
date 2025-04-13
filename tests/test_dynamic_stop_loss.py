import unittest
from genmarket.strategy_plugins.dynamic_stop_loss import DynamicStopLossStrategy

class TestDynamicStopLossStrategy(unittest.TestCase):
    def test_trailing_stop_loss(self):
        strategy = DynamicStopLossStrategy(stop_loss_threshold=0.05, trailing_stop=True)
        lob_result = [102, 105, 103, 101, 97]

        # Generate orders
        actions = strategy.generate_orders(lob_result)

        # Verify the actions
        expected_actions = ["HOLD", "HOLD", "HOLD", "HOLD", "SELL"]
        self.assertEqual(actions, expected_actions)

    def test_fixed_stop_loss(self):
        strategy = DynamicStopLossStrategy(stop_loss_threshold=0.05, trailing_stop=False)
        lob_result = [102, 95, 97]

        # Generate orders
        actions = strategy.generate_orders(lob_result)

        # Verify the actions
        expected_actions = ["HOLD", "SELL", "SELL"]
        self.assertEqual(actions, expected_actions)

if __name__ == "__main__":
    unittest.main()