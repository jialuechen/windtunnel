from genmarket import GenMarket
import json

# Load config
with open("configs/market_crash_scenario.json") as f:
    config = json.load(f)

# Run simulation
gm = GenMarket(config)
metrics = gm.run()

# Output results
print("Simulation Complete!")
print("Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")