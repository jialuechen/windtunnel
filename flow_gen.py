class OrderFlowGenerator:
    def generate(self, regime):
        return [
            {"timestamp": 0, "price": 100, "size": 10, "side": "buy"},
            {"timestamp": 1, "price": 101, "size": 15, "side": "sell"},
        ]
