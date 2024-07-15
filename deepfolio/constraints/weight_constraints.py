import keras
import tensorflow as tf

class WeightConstraints(keras.constraints.Constraint):
    def __init__(self, lower_bound=0.0, upper_bound=1.0):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, w):
        return tf.clip_by_value(w, self.lower_bound, self.upper_bound)