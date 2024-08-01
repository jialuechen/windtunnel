import tensorflow as tf

class FactorModel(tf.keras.Model):
    def __init__(self, n_factors):
        super(FactorModel, self).__init__()
        self.factor_loadings = tf.keras.layers.Dense(n_factors, use_bias=False)
    
    def call(self, factor_returns):
        return self.factor_loadings(factor_returns)
    
    def get_factor_exposures(self):
        return self.factor_loadings.weights[0]
