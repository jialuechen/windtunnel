import keras
import tensorflow as tf
from .base import BaseModel

class DistributionallyRobustCVaR(BaseModel):
    def __init__(self, alpha=0.05, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.output_layer = keras.layers.Dense(1, activation='softmax')

    def call(self, inputs):
        # Simplified implementation of DR-CVaR
        returns = inputs
        sorted_returns = tf.sort(returns, axis=1)
        var = sorted_returns[:, int(self.alpha * tf.shape(returns)[1])]
        cvar = tf.reduce_mean(tf.where(returns <= var[:, tf.newaxis], returns, 0), axis=1)
        
        # Add distributional robustness (simplified)
        robust_cvar = cvar - self.epsilon * tf.math.reduce_std(returns, axis=1)
        
        return self.output_layer(robust_cvar[:, tf.newaxis])