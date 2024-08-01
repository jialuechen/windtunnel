import tensorflow as tf

class DiffOptPortfolio(tf.keras.Model):
    def __init__(self, input_dim, n_assets, hidden_dim):
        super(DiffOptPortfolio, self).__init__()
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.mu_predictor = tf.keras.layers.Dense(n_assets)
        self.sigma_predictor = tf.keras.layers.Dense(n_assets * n_assets)
        self.qp_layer = QPLayer(n_assets)
    
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        mu = self.mu_predictor(features)
        sigma = tf.reshape(self.sigma_predictor(features), (-1, mu.shape[1], mu.shape[1]))
        return self.qp_layer([mu, sigma])

class QPLayer(tf.keras.layers.Layer):
    def __init__(self, n_assets):
        super(QPLayer, self).__init__()
        self.n_assets = n_assets
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.n_assets, self.n_assets),
                                 initializer='random_normal',
                                 trainable=True)
    
    def call(self, inputs):
        mu, Sigma = inputs
        def objective_fn(w):
            return tf.reduce_sum(tf.matmul(w, tf.matmul(Sigma, w, transpose_b=True))) - tf.reduce_sum(mu * w)
        
        constraints = [{'type': 'eq', 'fun': lambda w: tf.reduce_sum(w) - 1.0}]
        bounds = [(0, None) for _ in range(self.n_assets)]
        
        initial_w = tf.ones((self.n_assets,)) / self.n_assets
        
        optimized_w = tf.py_function(
            lambda: scipy.optimize.minimize(
                objective_fn, initial_w, method='SLSQP', 
                constraints=constraints, bounds=bounds
            ).x,
            [], tf.float32
        )
        
        return optimized_w