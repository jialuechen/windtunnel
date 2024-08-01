import tensorflow as tf
import cvxpy as cp
import numpy as np

class MultiPeriodOptimizer(tf.keras.layers.Layer):
    def __init__(self, n_assets, n_periods, transaction_cost=0.001):
        super(MultiPeriodOptimizer, self).__init__()
        self.n_assets = n_assets
        self.n_periods = n_periods
        self.transaction_cost = transaction_cost

    def call(self, inputs):
        mu_sequence, Sigma_sequence = inputs
        
        def solve_multi_period_qp(mu_sequence, Sigma_sequence):
            w = cp.Variable((self.n_periods, self.n_assets))
            risk_aversion = cp.Parameter(nonneg=True)
            
            objective = 0
            constraints = [cp.sum(w[0]) == 1, w[0] >= 0]
            
            for t in range(self.n_periods):
                objective += mu_sequence[t] @ w[t] - risk_aversion * cp.quad_form(w[t], Sigma_sequence[t])
                if t > 0:
                    objective -= self.transaction_cost * cp.sum(cp.abs(w[t] - w[t-1]))
                    constraints += [cp.sum(w[t]) == 1, w[t] >= 0]
            
            prob = cp.Problem(cp.Maximize(objective), constraints)
            risk_aversion.value = 1.0  # Initial value for risk aversion
            
            try:
                prob.solve(solver=cp.SCS)
                if prob.status != cp.OPTIMAL:
                    raise ValueError('Optimization problem not solved optimally')
                return w.value
            except:
                # Fallback to equal-weight portfolio if optimization fails
                return np.ones((self.n_periods, self.n_assets)) / self.n_assets

        optimized_w = tf.py_function(
            func=solve_multi_period_qp,
            inp=[mu_sequence, Sigma_sequence],
            Tout=tf.float32
        )
        
        return optimized_w

class MultiPeriodDiffOptPortfolio(tf.keras.Model):
    def __init__(self, input_dim, n_assets, n_periods, hidden_dim, transaction_cost=0.001):
        super(MultiPeriodDiffOptPortfolio, self).__init__()
        self.n_periods = n_periods
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.mu_predictor = tf.keras.layers.Dense(n_assets)
        self.sigma_predictor = tf.keras.layers.Dense(n_assets * n_assets)
        self.multi_period_optimizer = MultiPeriodOptimizer(n_assets, n_periods, transaction_cost)
    
    def call(self, inputs):
        features_sequence = tf.unstack(inputs, axis=1)  # Unstack along time dimension
        mu_sequence = []
        sigma_sequence = []
        
        for features in features_sequence:
            extracted_features = self.feature_extractor(features)
            mu = self.mu_predictor(extracted_features)
            sigma = tf.reshape(self.sigma_predictor(extracted_features), (-1, mu.shape[1], mu.shape[1]))
            mu_sequence.append(mu)
            sigma_sequence.append(sigma)
        
        mu_sequence = tf.stack(mu_sequence, axis=1)
        sigma_sequence = tf.stack(sigma_sequence, axis=1)
        
        return self.multi_period_optimizer([mu_sequence, sigma_sequence])