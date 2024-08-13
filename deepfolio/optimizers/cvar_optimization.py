import tensorflow as tf
import cvxpy as cp
import numpy as np

class CVaROptimizer(tf.keras.layers.Layer):
    def __init__(self, n_assets, n_scenarios, alpha=0.95):
        super(CVaROptimizer, self).__init__()
        self.n_assets = n_assets
        self.n_scenarios = n_scenarios
        self.alpha = alpha

    def call(self, inputs):
        returns_scenarios, = inputs
        
        def solve_cvar_optimization(returns_scenarios):
            w = cp.Variable(self.n_assets)
            aux_var = cp.Variable(1)
            slack_vars = cp.Variable(self.n_scenarios)
            
            portfolio_returns = returns_scenarios @ w
            objective = aux_var - (1 / (self.n_scenarios * (1 - self.alpha))) * cp.sum(slack_vars)
            
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                slack_vars >= 0,
                slack_vars >= -portfolio_returns - aux_var
            ]
            
            prob = cp.Problem(cp.Maximize(objective), constraints)
            
            try:
                prob.solve(solver=cp.SCS)
                if prob.status != cp.OPTIMAL:
                    raise ValueError('Optimization problem not solved optimally')
                return w.value
            except:
                # Fallback to equal-weight portfolio if optimization fails
                return np.ones(self.n_assets) / self.n_assets

        optimized_w = tf.py_function(
            func=solve_cvar_optimization,
            inp=[returns_scenarios],
            Tout=tf.float32
        )
        
        return optimized_w

class CVaRDiffOptPortfolio(tf.keras.Model):
    def __init__(self, input_dim, n_assets, n_scenarios, hidden_dim, alpha=0.95):
        super(CVaRDiffOptPortfolio, self).__init__()
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.returns_scenarios_generator = tf.keras.layers.Dense(n_assets * n_scenarios)
        self.cvar_optimizer = CVaROptimizer(n_assets, n_scenarios, alpha)
    
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        returns_scenarios = tf.reshape(self.returns_scenarios_generator(features), (-1, self.n_scenarios, self.n_assets))
        return self.cvar_optimizer([returns_scenarios])