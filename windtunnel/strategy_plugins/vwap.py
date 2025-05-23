class VWAPStrategy:
    def __init__(self, target_volume, time_horizon):
        self.target_volume = target_volume
        self.time_horizon = time_horizon

    def generate_orders(self, market_data):
        slice_volume = self.target_volume // self.time_horizon
        return [
            {"timestamp": t, "price": 100 + t % 2, "size": slice_volume, "side": "buy"}
            for t in range(self.time_horizon)
        ]