import keras
import tensorflow as tf

class TurnoverConstraint(keras.constraints.Constraint):
    def __init__(self, previous_weights, max_turnover):
        self.previous_weights = previous_weights
        self.max_turnover = max_turnover

    def __call__(self, w):
        turnover = tf.reduce_sum(tf.abs(w - self.previous_weights))
        if turnover > self.max_turnover:
            return self.previous_weights + (w - self.previous_weights) * (self.max_turnover / turnover)
        return w