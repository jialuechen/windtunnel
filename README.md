# GenMarket : Market Simulation by Generation

**GenMarket** is a modular, multi-layer generative market simulation framework designed to evaluate trading strategies and market impact under synthetic yet realistic conditions.

It combines macro regime generation, high-frequency order flow simulation, LOB microstructure modeling, and plug-in execution strategies (e.g. VWAP, TWAP), supporting both controlled experimentation and generative scenario construction.

---

## 🚀 Features

- **Multi-layer simulation**  
  Top-down generation from macro regimes → order flow → limit order book (LOB).

- **Scenario generation**  
  Uses VAE/Diffusion model stubs to create synthetic market conditions beyond historical replay.

- **Order flow modeling**  
  Autoregressive event-level generator simulates realistic buy/sell order streams.

- **LOB simulation engine**  
  Matches orders with queue logic and customizable liquidity depth.

- **Built-in execution strategies**  
  Plug-and-play VWAP and TWAP strategies for benchmark testing.

- **Market impact evaluation**  
  Measures execution price, slippage, and volume impact.

- **Visualization**  
  Plot order flow, execution trajectory, and size dynamics.

---

## 🧱 Installation

```bash
pip install --upgrade genmarket
```

---

## 📄 Example Config (`configs/market_crash_scenario.json`)

```json
{
  "regime": "volatile",
  "volatility": 0.35,
  "liquidity": "low",
  "lob": {
    "levels": 5,
    "latency_ms": 10
  },
  "strategy": {
    "type": "vwap",
    "params": {
      "target_volume": 100,
      "time_horizon": 10
    }
  }
}
```

---

## 🧪 Run a Simulation

```bash
python examples/run_simulation.py
```

```python
from genmarket import GenMarket
from genmarket.plotting import plot_executions
import json

with open("configs/market_crash_scenario.json") as f:
    config = json.load(f)

gm = GenMarket(config)
metrics = gm.run()

print("Simulation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Visualize execution result
plot_executions(metrics.get("executions", []))
```

---

## 🧩 Strategy Plugins

- `VWAPStrategy` – Volume-weighted execution across time horizon  
- `TWAPStrategy` – Time-weighted slices across fixed intervals  

You can create your own strategies and add them under `genmarket/strategy_plugins/`.

---

## 🔬 Applications

- Strategy benchmarking under stress scenarios  
- Execution cost modeling and market impact analysis  
- Synthetic data generation for pre-trade analytics  
- Reinforcement learning environment (future support)

---

## 📅 Roadmap

- [ ] Integrate pretrained diffusion/VAE market generators  
- [ ] Add LLM-driven natural language scenario parser  
- [ ] Extend to multi-asset & cross-venue simulation  
- [ ] Add dashboard interface for real-time simulation control

---

## 📜 License

MIT License. See `LICENSE`.

---

## 📚 Citation

```
@misc{genmarket2025,
  title   = {GenMarket: A Multi-Layer Generative Market Simulation Framework},
  author  = {Jialue Chen},
  year    = {2025},
  note    = {https://github.com/jialuechen/genmarket}
}
```
