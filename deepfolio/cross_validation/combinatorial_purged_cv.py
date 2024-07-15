import keras
import tensorflow as tf
import itertools

class CombinatorialPurgedCV(keras.layers.Layer):
    def __init__(self, n_splits, purge_period, **kwargs):
        super().__init__(**kwargs)
        self.n_splits = n_splits
        self.purge_period = purge_period

    def call(self, inputs):
        n_samples = tf.shape(inputs)[1]
        indices = tf.range(n_samples)
        
        split_points = tf.linspace(0.0, tf.cast(n_samples, tf.float32), self.n_splits + 1)
        split_points = tf.cast(split_points, tf.int32)
        
        all_splits = []
        for train_indices in itertools.combinations(range(1, self.n_splits), self.n_splits - 1):
            train_indices = list(train_indices)
            test_index = list(set(range(1, self.n_splits)) - set(train_indices))[0]
            
            train_mask = tf.zeros(n_samples, dtype=tf.bool)
            for i in train_indices:
                train_mask = tf.logical_or(train_mask, (indices >= split_points[i-1]) & (indices < split_points[i]))
            
            test_mask = (indices >= split_points[test_index-1]) & (indices < split_points[test_index])
            
            # Apply purging
            purge_mask = (indices < split_points[test_index-1] - self.purge_period) | (indices >= split_points[test_index])
            train_mask = tf.logical_and(train_mask, purge_mask)
            
            all_splits.append((train_mask, test_mask))
        
        return all_splits