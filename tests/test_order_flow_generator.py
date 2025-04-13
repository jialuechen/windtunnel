import unittest
from genmarket.order_flow_generator import OrderFlowGenerator
from genmarket.lob_simulator import LOBSimulator

class TestParallelSimulation(unittest.TestCase):
    def test_order_flow_parallel(self):
        order_flow_params = [
            {"base_price": 100, "steps": 10},
            {"base_price": 200, "steps": 10},
        ]
        generator = OrderFlowGenerator(flow_params={})
        results = generator.generate_order_flow_parallel(order_flow_params, num_threads=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], [100 + i for i in range(10)])
        self.assertEqual(results[1], [200 + i for i in range(10)])

    def test_lob_simulation_parallel(self):
        lob_params = [
            {"base_bid": 100, "base_ask": 101},
            {"base_bid": 200, "base_ask": 201},
        ]
        simulator = LOBSimulator(lob_params={})
        results = simulator.simulate_lob_parallel(lob_params, num_threads=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["bids"], [100 - i for i in range(5)])
        self.assertEqual(results[0]["asks"], [101 + i for i in range(5)])

if __name__ == "__main__":
    unittest.main()