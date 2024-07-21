import websocket
import json

class RealTimeData:
    def __init__(self, socket_url):
        self.socket_url = socket_url

    def on_message(self, ws, message):
        data = json.loads(message)
        # Process real-time data (e.g., update portfolio)
        print(data)

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws):
        print("### closed ###")

    def on_open(self, ws):
        print("### opened ###")

    def run(self):
        ws = websocket.WebSocketApp(self.socket_url, 
                                    on_message=self.on_message, 
                                    on_error=self.on_error, 
                                    on_close=self.on_close)
        ws.on_open = self.on_open
        ws.run_forever()

if __name__ == "__main__":
    socket_url = "wss://cloud-sse.iexapis.com/stable/stocksUSNoUTP?token=YOUR_IEX_CLOUD_TOKEN"
    real_time_data = RealTimeData(socket_url)
    real_time_data.run()
