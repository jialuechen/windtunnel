import keras
import tensorflow as tf

class NonDominatedSelectionLayer(keras.layers.Layer):
    def __init__(self, risk_measure, **kwargs):
        super().__init__(**kwargs)
        self.risk_measure = risk_measure

    def call(self, inputs):
        mean_returns = tf.reduce_mean(inputs, axis=1)
        risks = self.risk_measure(inputs)
        
        # This is a simplified version and may not be efficient for large datasets
        # A more efficient implementation would be needed for production use
        dominated = tf.zeros_like(mean_returns, dtype=tf.bool)
        for i in range(tf.shape(mean_returns)[1]):
            for j in range(tf.shape(mean_returns)[1]):
                if i != j:
                    dominated = tf.logical_or(
                        dominated,
                        tf.logical_and(
                            mean_returns[:, j] > mean_returns[:, i],
                            risks[:, j] < risks[:, i]
                        )
                    )
        
        return tf.cast(tf.logical_not(dominated), tf.float32)