import alpaca_trade_api as tradeapi

class AutomatedTrading:
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url)

    def place_trade(self, symbol, qty, side):
        self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )

if __name__ == "__main__":
    api_key = 'APCA-API-KEY-ID'
    secret_key = 'APCA-API-SECRET-KEY'
    base_url = 'https://paper-api.alpaca.markets'

    trader = AutomatedTrading(api_key, secret_key, base_url)
    trader.place_trade('AAPL', 10, 'buy')
