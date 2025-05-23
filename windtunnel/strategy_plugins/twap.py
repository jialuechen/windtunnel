class TWAPStrategy:
    def __init__(self, target_volume, intervals):
        self.target_volume = target_volume
        self.intervals = intervals

    def generate_orders(self, market_data):
        slice_volume = self.target_volume // self.intervals
        return [
            {"timestamp": t * 10, "price": 100, "size": slice_volume, "side": "buy"}
            for t in range(self.intervals)
        ]