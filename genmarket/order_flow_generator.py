from concurrent.futures import ThreadPoolExecutor

class OrderFlowGenerator:
    def __init__(self, flow_params):
        """
        Initialize the order flow generator.

        Args:
            flow_params (dict): Parameters for generating order flow.
        """
        self.flow_params = flow_params

    def generate_order_flow(self, param):
        """
        Generate order flow for a single parameter set.

        Args:
            param (dict): Parameters for generating a single order flow.

        Returns:
            list: Simulated order flow data.
        """
        # Simulate order flow (replace with actual logic)
        return [param.get("base_price", 100) + i for i in range(param.get("steps", 10))]

    def generate_order_flow_parallel(self, param_list, num_threads=4):
        """
        Generate order flow in parallel for multiple parameter sets.

        Args:
            param_list (list): List of parameter dictionaries for each order flow.
            num_threads (int): Number of threads to use for parallel processing.

        Returns:
            list: List of simulated order flow data for each parameter set.
        """
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.generate_order_flow, param_list))
        return results