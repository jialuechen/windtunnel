import keras
import tensorflow as tf

class TrackingErrorConstraint(keras.constraints.Constraint):
    def __init__(self, benchmark_weights, max_tracking_error):
        self.benchmark_weights = benchmark_weights
        self.max_tracking_error = max_tracking_error

    def __call__(self, w):
        tracking_error = tf.sqrt(tf.reduce_sum(tf.square(w - self.benchmark_weights)))
        scale = tf.minimum(1.0, self.max_tracking_error / (tracking_error + 1e-8))
        return self.benchmark_weights + scale * (w - self.benchmark_weights)