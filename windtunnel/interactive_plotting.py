import plotly.graph_objects as go

class InteractivePlotting:
    @staticmethod
    def plot_execution_prices(prices, timestamps):
        """
        Plot execution prices over time.

        Args:
            prices (list): List of execution prices.
            timestamps (list): List of timestamps corresponding to the prices.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=prices, mode='lines+markers', name='Execution Prices'))
        fig.update_layout(
            title='Execution Prices Over Time',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark'
        )
        fig.show()

    @staticmethod
    def plot_order_book_depth(bid_depth, ask_depth, levels):
        """
        Plot order book depth (bids and asks).

        Args:
            bid_depth (list): List of bid depths at each price level.
            ask_depth (list): List of ask depths at each price level.
            levels (list): List of price levels.
        """
        fig = go.Figure()
        fig.add_trace(go.Bar(x=levels, y=bid_depth, name='Bid Depth', marker_color='green'))
        fig.add_trace(go.Bar(x=levels, y=ask_depth, name='Ask Depth', marker_color='red'))
        fig.update_layout(
            title='Order Book Depth',
            xaxis_title='Price Levels',
            yaxis_title='Depth',
            barmode='overlay',
            template='plotly_dark'
        )
        fig.show()