import keras
import tensorflow as tf

class SelectKExtremes(keras.layers.Layer):
    def __init__(self, k, select_best=True, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.select_best = select_best

    def call(self, inputs):
        mean_returns = tf.reduce_mean(inputs, axis=1)
        if self.select_best:
            _, indices = tf.nn.top_k(mean_returns, k=self.k)
        else:
            _, indices = tf.nn.top_k(-mean_returns, k=self.k)
        
        mask = tf.reduce_sum(tf.one_hot(indices, depth=tf.shape(inputs)[2]), axis=1)
        return mask