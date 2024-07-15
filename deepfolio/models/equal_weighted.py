import keras
import tensorflow as tf
from .base import BaseModel

class EqualWeighted(BaseModel):
    def call(self, inputs):
        n_assets = tf.shape(inputs)[2]
        return tf.ones_like(inputs[:, 0, :]) / tf.cast(n_assets, tf.float32)