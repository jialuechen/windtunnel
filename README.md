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

### Deep Learning Models
- Transformer
- RNN

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

### Optimization with Transformer
```python
from deepfolio.models.transformer import Transformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Model parameters
n_feature = 5  # Number of features per asset
n_assets = 10  # Number of assets
n_timestep = 30  # Number of time steps
n_layer = 3  # Number of Transformer layers
n_head = 8  # Number of attention heads
n_hidden = 64  # Number of hidden units
n_dropout = 0.1  # Dropout rate
batch_size = 32
epochs = 50
lb = 0.0  # Lower bound for asset weights
ub = 1.0  # Upper bound for asset weights

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# Get the first 10 stocks of S&P 500 as an example
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].tolist()[:n_assets]

# Download stock data
stock_data = get_stock_data(tickers, '2010-01-01', '2023-01-01')

# Calculate daily returns
returns = stock_data.pct_change().dropna()

def calculate_features(returns):
    features = pd.DataFrame()
    for ticker in returns.columns:
        # Calculate 5-day, 10-day, and 20-day moving averages
        features[f'{ticker}_MA5'] = returns[ticker].rolling(window=5).mean()
        features[f'{ticker}_MA10'] = returns[ticker].rolling(window=10).mean()
        features[f'{ticker}_MA20'] = returns[ticker].rolling(window=20).mean()
        # Calculate 5-day, 10-day, and 20-day volatility
        features[f'{ticker}_VOL5'] = returns[ticker].rolling(window=5).std()
        features[f'{ticker}_VOL10'] = returns[ticker].rolling(window=10).std()
        features[f'{ticker}_VOL20'] = returns[ticker].rolling(window=20).std()
    return features.dropna()

features = calculate_features(returns)

# Prepare input data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

X = []
y = []
for i in range(len(scaled_features) - n_timestep):
    X.append(scaled_features[i:i+n_timestep])
    y.append(returns.iloc[i+n_timestep].values)

X = np.array(X)
y = np.array(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom loss function: negative Sharpe ratio
def negative_sharpe_ratio(y_true, y_pred):
    returns = tf.reduce_sum(y_true * y_pred, axis=1)
    expected_return = tf.reduce_mean(returns)
    stddev = tf.math.reduce_std(returns)
    return -expected_return / (stddev + 1e-6)  # Add small value to avoid division by zero

# Create and compile the model
model = Transformer(n_feature * n_assets, n_timestep, n_layer, n_head, n_hidden, n_dropout, n_assets, lb, ub)
model.compile(optimizer='adam', loss=negative_sharpe_ratio)

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")

# Make predictions using the model
predictions = model.predict(X_test)

# Calculate Sharpe ratio on the test set
test_returns = np.sum(y_test * predictions, axis=1)
sharpe_ratio = np.mean(test_returns) / np.std(test_returns)
print(f"Sharpe Ratio on test set: {sharpe_ratio}")

# Visualize results
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize asset allocation for the last time step
plt.figure(figsize=(10, 5))
plt.bar(tickers, predictions[-1])
plt.title('Asset Allocation for Last Time Step')
plt.xlabel('Assets')
plt.ylabel('Weight')
plt.xticks(rotation=45)
plt.show()
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
