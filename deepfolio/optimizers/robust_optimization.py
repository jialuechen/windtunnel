import tensorflow as tf
import cvxpy as cp
import numpy as np

class RobustMeanVarianceOptimizer(tf.keras.layers.Layer):
    def __init__(self, n_assets, uncertainty_budget=0.1):
        super(RobustMeanVarianceOptimizer, self).__init__()
        self.n_assets = n_assets
        self.uncertainty_budget = uncertainty_budget

    def call(self, inputs):
        mu, Sigma = inputs
        
        def solve_robust_qp(mu, Sigma):
            w = cp.Variable(self.n_assets)
            kappa = cp.Parameter(nonneg=True)
            
            obj = cp.Maximize(mu @ w - kappa * cp.quad_form(w, Sigma))
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                cp.norm(cp.sqrt(Sigma) @ w) <= self.uncertainty_budget
            ]
            
            prob = cp.Problem(obj, constraints)
            kappa.value = 1.0  # Initial value for risk aversion
            
            try:
                prob.solve(solver=cp.SCS)
                if prob.status != cp.OPTIMAL:
                    raise ValueError('Optimization problem not solved optimally')
                return w.value
            except:
                # Fallback to equal-weight portfolio if optimization fails
                return np.ones(self.n_assets) / self.n_assets

        optimized_w = tf.py_function(
            func=solve_robust_qp,
            inp=[mu, Sigma],
            Tout=tf.float32
        )
        
        return optimized_w

class RobustDiffOptPortfolio(tf.keras.Model):
    def __init__(self, input_dim, n_assets, hidden_dim, uncertainty_budget=0.1):
        super(RobustDiffOptPortfolio, self).__init__()
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.mu_predictor = tf.keras.layers.Dense(n_assets)
        self.sigma_predictor = tf.keras.layers.Dense(n_assets * n_assets)
        self.robust_optimizer = RobustMeanVarianceOptimizer(n_assets, uncertainty_budget)
    
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        mu = self.mu_predictor(features)
        sigma = tf.reshape(self.sigma_predictor(features), (-1, mu.shape[1], mu.shape[1]))
        return self.robust_optimizer([mu, sigma])
