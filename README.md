# deepfolio: Portfolio Optimization with Machine Learning

[![PyPI version](https://badge.fury.io/py/deepfolio.svg)](https://badge.fury.io/py/deepfolio)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/deepfolio.svg)](https://pypi.org/project/deepfolio/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red)](https://keras.io/)

**deepfolio** is a Python library for portfolio optimization built on top of Keras 3. It offers a unified interface and tools compatible with Keras to build, fine-tune, and cross-validate portfolio models.

## Installation

Install the package using pip:

```bash
pip install --upgrade deepfolio
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

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This package leverages the power of Keras 3 for efficient portfolio optimization.
- Thanks to the financial machine learning community for inspiring many of the implemented methods.
