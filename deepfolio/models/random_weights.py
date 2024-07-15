import keras
import tensorflow as tf
from .base import BaseModel

class RandomWeights(BaseModel):
    def call(self, inputs):
        n_assets = tf.shape(inputs)[2]
        random_weights = tf.random.uniform(shape=(tf.shape(inputs)[0], n_assets))
        return random_weights / tf.reduce_sum(random_weights, axis=1, keepdims=True)