import keras
import tensorflow as tf

class EmpiricalReturnsEstimator(keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

class EquilibriumEstimator(keras.layers.Layer):
    def __init__(self, risk_aversion=1.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion

    def call(self, inputs):
        cov = tfp.stats.covariance(inputs)
        market_weights = tf.ones_like(inputs[:, 0, :]) / tf.shape(inputs)[2]
        return self.risk_aversion * tf.linalg.matvec(cov, market_weights)

class ShrinkageEstimator(keras.layers.Layer):
    def __init__(self, shrinkage_factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.shrinkage_factor = shrinkage_factor

    def call(self, inputs):
        sample_mean = tf.reduce_mean(inputs, axis=1)
        grand_mean = tf.reduce_mean(sample_mean, axis=1, keepdims=True)
        return self.shrinkage_factor * grand_mean + (1 - self.shrinkage_factor) * sample_mean