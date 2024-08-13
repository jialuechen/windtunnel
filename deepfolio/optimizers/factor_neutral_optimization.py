import tensorflow as tf
import cvxpy as cp
import numpy as np

class FactorNeutralOptimizer(tf.keras.layers.Layer):
    def __init__(self, n_assets, n_factors, factor_exposure_bounds=(-0.1, 0.1)):
        super(FactorNeutralOptimizer, self).__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.factor_exposure_bounds = factor_exposure_bounds

    def call(self, inputs):
        mu, Sigma, factor_exposures = inputs
        
        def solve_factor_neutral_qp(mu, Sigma, factor_exposures):
            w = cp.Variable(self.n_assets)
            risk_aversion = cp.Parameter(nonneg=True)
            
            objective = mu @ w - risk_aversion * cp.quad_form(w, Sigma)
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                factor_exposures @ w >= self.factor_exposure_bounds[0],
                factor_exposures @ w <= self.factor_exposure_bounds[1]
            ]
            
            prob = cp.Problem(cp.Maximize(objective), constraints)
            risk_aversion.value = 1.0  # Initial value for risk aversion
            
            try:
                prob.solve(solver=cp.SCS)
                if prob.status != cp.OPTIMAL:
                    raise ValueError('Optimization problem not solved optimally')
                return w.value
            except:
                # Fallback to equal-weight portfolio if optimization fails
                return np.ones(self.n_assets) / self.n_assets

        optimized_w = tf.py_function(
            func=solve_factor_neutral_qp,
            inp=[mu, Sigma, factor_exposures],
            Tout=tf.float32
        )
        
        return optimized_w

class FactorNeutralDiffOptPortfolio(tf.keras.Model):
    def __init__(self, input_dim, n_assets, n_factors, hidden_dim, factor_exposure_bounds=(-0.1, 0.1)):
        super(FactorNeutralDiffOptPortfolio, self).__init__()
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.mu_predictor = tf.keras.layers.Dense(n_assets)
        self.sigma_predictor = tf.keras.layers.Dense(n_assets * n_assets)
        self.factor_exposures_predictor = tf.keras.layers.Dense(n_assets * n_factors)
        self.factor_neutral_optimizer = FactorNeutralOptimizer(n_assets, n_factors, factor_exposure_bounds)
    
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        mu = self.mu_predictor(features)
        sigma = tf.reshape(self.sigma_predictor(features), (-1, mu.shape[1], mu.shape[1]))
        factor_exposures = tf.reshape(self.factor_exposures_predictor(features), (-1, mu.shape[1], self.n_factors))
        return self.factor_neutral_optimizer([mu, sigma, factor_exposures])