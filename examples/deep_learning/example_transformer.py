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
