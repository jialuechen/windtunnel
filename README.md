<div align=center>
<img src="assets/deepfolio.png" width="45%" loc>
</div>
<div align=center>

# DeepFolio: Real-time Portfolio Optimization with Deep Learning

![PyPI - Version](https://img.shields.io/pypi/v/deepfolio)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
![Python versions](https://img.shields.io/badge/python-3.12%2B-green)
![PyPI downloads](https://img.shields.io/pypi/dm/deepfolio)
[![Keras](https://img.shields.io/badge/Keras-3.x-red)](https://keras.io/)

</div>

**DeepFolio** is a Python library for portfolio optimization built on top of Keras 3 and TensorFlow 2. It offers a unified interface and tools compatible with Keras to build, fine-tune, and cross-validate portfolio models.

## Installation

Install the package using pip:

```bash
pip install deepfolio
```

## Quick Start

Here's a simple example to get you started with deepfolio:

```python
import numpy as np
from deepfolio.models import MeanRisk
from deepfolio.estimators import EmpiricalReturnsEstimator
from deepfolio.risk_measures import Variance

# Generate sample data
returns = np.random.randn(100, 10)  # 100 time steps, 10 assets

# Initialize estimators and risk measure
returns_estimator = EmpiricalReturnsEstimator()
risk_measure = Variance()

# Create and fit the model
model = MeanRisk(returns_estimator=returns_estimator, risk_measure=risk_measure)
model.fit(returns)

# Get optimal weights
optimal_weights = model.predict(returns)
print("Optimal portfolio weights:", optimal_weights)
```


## Available Models and Features

### Automated Trading

DeepFolio now supports automated trading through integration with the Alpaca API. This feature allows users to:

Place Trades: Automatically place buy/sell orders based on portfolio optimization results.
Execution Logic: Execute trades with customizable order parameters.
Example usage:
```python
from deepfolio.models.automated_trading import AutomatedTrading

api_key = 'APCA-API-KEY-ID'
secret_key = 'APCA-API-SECRET-KEY'
base_url = 'https://paper-api.alpaca.markets'

trader = AutomatedTrading(api_key, secret_key, base_url)
trader.place_trade('AAPL', 10, 'buy')
```

### Real-Time Data Integration
DeepFolio now includes real-time data integration using WebSocket. This feature enables:

Real-Time Market Data: Receive and process streaming market data for dynamic portfolio adjustments.
Data Feeds: Integration with IEX Cloud for real-time data streaming.
Example usage:

```python
from deepfolio.data.real_time_data import RealTimeData

socket_url = "wss://cloud-sse.iexapis.com/stable/stocksUSNoUTP?token=YOUR_IEX_CLOUD_TOKEN"
real_time_data = RealTimeData(socket_url)
real_time_data.run()

```

### Portfolio Optimization
- Naive: Equal-Weighted, Random (Dirichlet)
- Convex: Mean-Risk, Distributionally Robust CVaR
- Clustering: Hierarchical Risk Parity, Hierarchical Equal Risk Contribution, Nested Clusters Optimization

### Expected Returns Estimator
- Empirical
- Equilibrium
- Shrinkage

### Distance Estimator
- Pearson Distance
- Kendall Distance
- Variation of Information

### Pre-Selection Transformer
- Non-Dominated Selection
- Select K Extremes (Best or Worst)
- Drop Highly Correlated Assets

### Risk Measures
- Variance
- Semi-Variance
- Mean Absolute Deviation
- Skew
- Kurtosis

### Cross-Validation and Model Selection
- Walk Forward
- Combinatorial Purged Cross-Validation

### Optimization Features
- Minimize Risk
- Transaction Costs
- L1 and L2 Regularization
- Weight Constraints
- Tracking Error Constraints
- Turnover Constraints

## Examples

### Using Hierarchical Risk Parity

```python
from deepfolio.models import HierarchicalRiskParity
from deepfolio.estimators import EmpiricalReturnsEstimator
from deepfolio.distance import PearsonDistance

returns = np.random.randn(200, 20)  # 200 time steps, 20 assets

model = HierarchicalRiskParity(
    returns_estimator=EmpiricalReturnsEstimator(),
    distance_estimator=PearsonDistance()
)
model.fit(returns)
weights = model.predict(returns)
print("HRP portfolio weights:", weights)
```

### Cross-Validation

```python
from deepfolio.cross_validation import WalkForward
from deepfolio.models import MeanRisk
from deepfolio.risk_measures import SemiVariance

cv = WalkForward(n_splits=5, test_size=20)
model = MeanRisk(risk_measure=SemiVariance())

for train_index, test_index in cv.split(returns):
    train_returns, test_returns = returns[train_index], returns[test_index]
    model.fit(train_returns)
    weights = model.predict(test_returns)
    # Evaluate performance...
```

## Documentation

For full documentation, please visit our [documentation site](https://deepfolio.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the BSD-2-Clause License- see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This package leverages the power of Keras 3 for efficient portfolio optimization.
- Thanks to the financial machine learning community for inspiring many of the implemented methods.
