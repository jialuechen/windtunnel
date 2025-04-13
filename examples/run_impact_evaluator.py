from genmarket.impact_evaluator import ImpactEvaluator

# Example data
expected_price = 100.0
executed_price = 102.0
executed_prices = [102, 103, 101]
market_prices = [100, 101, 100]
entry_price = 100.0
exit_price = 105.0
position_size = 10

# Calculate slippage
slippage = ImpactEvaluator.calculate_slippage(expected_price, executed_price)
print(f"Slippage: {slippage:.2f}%")

# Calculate impact cost
impact_cost = ImpactEvaluator.calculate_impact_cost(executed_prices, market_prices)
print(f"Impact Cost: {impact_cost:.2f}")

# Calculate PnL
pnl = ImpactEvaluator.calculate_pnl(entry_price, exit_price, position_size)
print(f"PnL: {pnl:.2f}")