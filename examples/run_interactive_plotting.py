from genmarket.interactive_plotting import InteractivePlotting

# Example data for execution prices
timestamps = ['2025-04-13 10:00', '2025-04-13 10:01', '2025-04-13 10:02', '2025-04-13 10:03']
execution_prices = [100, 102, 101, 103]

# Plot execution prices
InteractivePlotting.plot_execution_prices(prices=execution_prices, timestamps=timestamps)

# Example data for order book depth
price_levels = [99, 100, 101, 102, 103]
bid_depth = [50, 70, 60, 40, 30]
ask_depth = [30, 40, 50, 60, 70]

# Plot order book depth
InteractivePlotting.plot_order_book_depth(bid_depth=bid_depth, ask_depth=ask_depth, levels=price_levels)