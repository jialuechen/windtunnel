import tensorflow as tf
import numpy as np
import cvxpy as cp

class BlackLittermanOptimizer(tf.keras.layers.Layer):
    def __init__(self, n_assets, risk_aversion=2.5, tau=0.05):
        super(BlackLittermanOptimizer, self).__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.tau = tau

    def call(self, inputs):
        market_caps, Sigma, views, view_confidences = inputs
        
        def black_litterman_optimization(market_caps, Sigma, views, view_confidences):
            # Calculate market equilibrium returns
            market_weights = market_caps / np.sum(market_caps)
            Pi = self.risk_aversion * Sigma @ market_weights

            # Prepare views
            P = np.eye(self.n_assets)[views[:, 0].astype(int)]
            Q = views[:, 1]
            Omega = np.diag(1 / view_confidences)

            # Black-Litterman formula
            BL_mean = np.linalg.inv(np.linalg.inv(self.tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P) @ \
                      (np.linalg.inv(self.tau * Sigma) @ Pi + P.T @ np.linalg.inv(Omega) @ Q)
            BL_cov = np.linalg.inv(np.linalg.inv(self.tau * Sigma) + P.T @ np.linalg.inv(Omega) @ P)

            # Optimization
            w = cp.Variable(self.n_assets)
            risk = cp.quad_form(w, BL_cov)
            ret = BL_mean.T @ w
            objective = cp.Maximize(ret - self.risk_aversion * risk)
            constraints = [cp.sum(w) == 1, w >= 0]

            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.SCS)
                if prob.status != cp.OPTIMAL:
                    raise ValueError('Optimization problem not solved optimally')
                return w.value
            except:
                return market_weights

        optimized_w = tf.py_function(
            func=black_litterman_optimization,
            inp=[market_caps, Sigma, views, view_confidences],
            Tout=tf.float32
        )
        
        return optimized_w

class BlackLittermanDiffOptPortfolio(tf.keras.Model):
    def __init__(self, input_dim, n_assets, hidden_dim, risk_aversion=2.5, tau=0.05):
        super(BlackLittermanDiffOptPortfolio, self).__init__()
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.market_cap_predictor = tf.keras.layers.Dense(n_assets)
        self.sigma_predictor = tf.keras.layers.Dense(n_assets * n_assets)
        self.views_predictor = tf.keras.layers.Dense(n_assets * 2)
        self.view_confidence_predictor = tf.keras.layers.Dense(n_assets)
        self.bl_optimizer = BlackLittermanOptimizer(n_assets, risk_aversion, tau)
    
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        market_caps = tf.exp(self.market_cap_predictor(features))  # Ensure positive market caps
        sigma = tf.reshape(self.sigma_predictor(features), (-1, self.n_assets, self.n_assets))
        views = tf.reshape(self.views_predictor(features), (-1, self.n_assets, 2))
        view_confidences = tf.exp(self.view_confidence_predictor(features))  # Ensure positive confidences
        return self.bl_optimizer([market_caps, sigma, views, view_confidences])