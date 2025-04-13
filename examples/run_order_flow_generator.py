from genmarket.order_flow_generator import OrderFlowGenerator
from genmarket.lob_simulator import LOBSimulator

# Example parameters for order flow generation
order_flow_params = [
    {"base_price": 100, "steps": 10},
    {"base_price": 200, "steps": 10},
    {"base_price": 300, "steps": 10},
]

# Example parameters for LOB simulation
lob_params = [
    {"base_bid": 100, "base_ask": 101},
    {"base_bid": 200, "base_ask": 201},
    {"base_bid": 300, "base_ask": 301},
]

# Initialize generators
order_flow_generator = OrderFlowGenerator(flow_params={})
lob_simulator = LOBSimulator(lob_params={})

# Generate order flows in parallel
order_flows = order_flow_generator.generate_order_flow_parallel(order_flow_params, num_threads=3)
print("Order Flows:", order_flows)

# Simulate LOBs in parallel
lobs = lob_simulator.simulate_lob_parallel(lob_params, num_threads=3)
print("LOBs:", lobs)