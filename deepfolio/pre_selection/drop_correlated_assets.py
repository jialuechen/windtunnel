import keras
import tensorflow as tf

class DropCorrelatedAssets(keras.layers.Layer):
    def __init__(self, threshold=0.95, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs):
        correlation_matrix = tfp.stats.correlation(inputs, sample_axis=1)
        n_assets = tf.shape(correlation_matrix)[1]
        
        # Create a mask for assets to keep
        mask = tf.ones((tf.shape(inputs)[0], n_assets), dtype=tf.float32)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                corr = correlation_matrix[:, i, j]
                condition = tf.abs(corr) > self.threshold
                # If assets are highly correlated, keep the first one (arbitrarily)
                mask = tf.where(condition[:, tf.newaxis], mask * tf.one_hot(i, n_assets), mask)
        
        return mask