class DynamicStopLossStrategy:
    def __init__(self, stop_loss_threshold, trailing_stop=True):
        """
        Initialize the dynamic stop loss strategy.

        Args:
            stop_loss_threshold (float): Percentage threshold for stop loss (e.g., 0.05 for 5%).
            trailing_stop (bool): Whether to use a trailing stop loss.
        """
        self.stop_loss_threshold = stop_loss_threshold
        self.trailing_stop = trailing_stop
        self.highest_price = None

    def generate_orders(self, lob_result):
        """
        Generate orders based on the stop loss strategy.

        Args:
            lob_result (list): A list of market prices (e.g., from LOB simulation).

        Returns:
            list: A list of actions ("SELL" or "HOLD") for each price in lob_result.
        """
        actions = []
        for current_price in lob_result:
            if self.trailing_stop:
                # Update the highest price if the current price exceeds it
                if self.highest_price is None or current_price > self.highest_price:
                    self.highest_price = current_price

                # Calculate the trailing stop loss price
                stop_loss_price = self.highest_price * (1 - self.stop_loss_threshold)
            else:
                # Fixed stop loss based on the first price
                stop_loss_price = lob_result[0] * (1 - self.stop_loss_threshold)

            # Check if the current price triggers the stop loss
            if current_price <= stop_loss_price:
                actions.append("SELL")
            else:
                actions.append("HOLD")

        return actions