import unittest
from genmarket.interactive_plotting import InteractivePlotting

class TestInteractivePlotting(unittest.TestCase):
    def test_plot_execution_prices(self):
        # Example data
        timestamps = ['2025-04-13 10:00', '2025-04-13 10:01', '2025-04-13 10:02']
        prices = [100, 102, 101]

        # Test if the function runs without errors
        try:
            InteractivePlotting.plot_execution_prices(prices=prices, timestamps=timestamps)
        except Exception as e:
            self.fail(f"plot_execution_prices raised an exception: {e}")

    def test_plot_order_book_depth(self):
        # Example data
        price_levels = [99, 100, 101]
        bid_depth = [50, 70, 60]
        ask_depth = [30, 40, 50]

        # Test if the function runs without errors
        try:
            InteractivePlotting.plot_order_book_depth(bid_depth=bid_depth, ask_depth=ask_depth, levels=price_levels)
        except Exception as e:
            self.fail(f"plot_order_book_depth raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()