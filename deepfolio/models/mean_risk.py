
import keras
import tensorflow as tf
from .base import BaseModel

class MeanRiskLayer(keras.layers.Layer):
    def __init__(self, risk_measure, **kwargs):
        super().__init__(**kwargs)
        self.risk_measure = risk_measure

    def call(self, inputs):
        mean_returns = tf.reduce_mean(inputs, axis=1)
        risks = self.risk_measure(inputs)
        return tf.stack([mean_returns, risks], axis=1)

class MeanRisk(BaseModel):
    def __init__(self, returns_estimator, risk_measure):
        super().__init__()
        self.returns_estimator = returns_estimator
        self.mean_risk_layer = MeanRiskLayer(risk_measure)
        self.output_layer = keras.layers.Dense(1, activation='softmax')

    def call(self, inputs):
        x = self.returns_estimator(inputs)
        x = self.mean_risk_layer(x)
        return self.output_layer(x)