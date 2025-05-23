
#### **`docs/modules/windtunnel.md`**
```md
# Module Description: `windtunnel`

## Core Modules
- **`RegimeGenerator`**: Generates market regimes.
- **`OrderFlowGenerator`**: Generates order flows.
- **`LOBSimulator`**: Simulates the limit order book.
- **`StrategyExecutor`**: Executes trading strategies.
- **`ImpactEvaluator`**: Evaluates market impact.

## Example
```python
from windtunnel import WindTunnel
import json

with open("configs/market_crash_scenario.json") as f:
    config = json.load(f)

gm = WindTunnel(config)
metrics = gm.run()

print("Simulation Metrics:", metrics)