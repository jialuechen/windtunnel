import keras
import tensorflow as tf

class Skew(keras.layers.Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=1, keepdims=True)
        z_scores = (inputs - mean) / std
        return tf.reduce_mean(tf.pow(z_scores, 3), axis=1)