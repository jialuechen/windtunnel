class ImpactEvaluator:
    @staticmethod
    def calculate_slippage(expected_price, executed_price):
        """
        Calculate slippage as the percentage difference between expected and executed prices.

        Args:
            expected_price (float): The expected price of the order.
            executed_price (float): The actual executed price of the order.

        Returns:
            float: The slippage percentage.
        """
        return (executed_price - expected_price) / expected_price * 100

    @staticmethod
    def calculate_impact_cost(executed_prices, market_prices):
        """
        Calculate the impact cost as the average difference between executed prices and market prices.

        Args:
            executed_prices (list): List of executed prices.
            market_prices (list): List of market prices at the time of execution.

        Returns:
            float: The average impact cost.
        """
        if len(executed_prices) != len(market_prices):
            raise ValueError("Executed prices and market prices must have the same length.")
        differences = [executed - market for executed, market in zip(executed_prices, market_prices)]
        return sum(differences) / len(differences)

    @staticmethod
    def calculate_pnl(entry_price, exit_price, position_size):
        """
        Calculate the profit and loss (PnL) for a position.

        Args:
            entry_price (float): The price at which the position was entered.
            exit_price (float): The price at which the position was exited.
            position_size (float): The size of the position.

        Returns:
            float: The profit or loss for the position.
        """
        return (exit_price - entry_price) * position_size