class LOBSimulator:
    def __init__(self, config):
        self.config = config
        self.book = []

    def simulate(self, orders):
        trades = []
        for order in orders:
            trades.append({"executed_price": order["price"], "size": order["size"]})
        return trades
