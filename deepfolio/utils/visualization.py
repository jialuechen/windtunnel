import matplotlib.pyplot as plt

def plot_portfolio_weights(weights, asset_names):
    plt.figure(figsize=(10, 6))
    plt.bar(asset_names, weights)
    plt.title('Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_returns(returns):
    plt.figure(figsize=(10, 6))
    plt.plot(returns.cumsum())
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.tight_layout()
    plt.show()