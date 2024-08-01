import tensorflow as tf
import numpy as np

class Backtester:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.results = None
    
    def run(self, initial_capital=10000):
        features = self.data['features']
        returns = self.data['returns']
        
        portfolio_values = [initial_capital]
        weights_history = []
        
        for i in range(len(features)):
            current_features = tf.convert_to_tensor(features[i:i+1], dtype=tf.float32)
            current_returns = returns[i]
            
            weights = self.model(current_features)
            weights = weights.numpy().flatten()
            
            portfolio_return = np.sum(weights * current_returns)
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
            weights_history.append(weights)
        
        self.results = {
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_values),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values)
        }
    
    def calculate_sharpe_ratio(self, portfolio_values):
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return np.sqrt(252) * np.mean(returns) / np.std(returns)
    
    def calculate_max_drawdown(self, portfolio_values):
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
    
    def get_results(self):
        return self.results