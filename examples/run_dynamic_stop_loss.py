from strategy_executor import StrategyExecutor

# Configuration for the dynamic stop loss strategy
config = {
    "type": "dynamic_stop_loss",
    "params": {
        "stop_loss_threshold": 0.05,  # 5% stop loss
        "trailing_stop": True        # Enable trailing stop
    }
}

# Initialize the strategy executor
executor = StrategyExecutor(config=config)

# Simulated LOB result (market prices)
lob_result = [102, 105, 103, 101, 97]  # Example market prices

# Run the strategy
result = executor.run(lob_result)

# Print the results
print("LOB Result:", lob_result)
print("Strategy Actions:", result["executions"])