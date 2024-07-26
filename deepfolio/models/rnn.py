import tensorflow as tf
import numpy as np

class RNN(tf.keras.Model):
    def __init__(self, n_feature, n_timestep, n_hidden, n_layer, n_dropout, n_output, lb, ub):
        super().__init__()
        self.n_feature = n_feature
        self.n_timestep = n_timestep
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.n_dropout = n_dropout
        self.n_output = n_output
        self.lb = lb
        self.ub = ub

        # LSTM layers
        self.lstm_layers = [
            tf.keras.layers.LSTM(n_hidden, return_sequences=True, dropout=n_dropout)
            for _ in range(n_layer - 1)
        ]
        self.lstm_layers.append(tf.keras.layers.LSTM(n_hidden, dropout=n_dropout))

        # Output layer
        self.out = tf.keras.layers.Dense(n_output)

    def call(self, x, training=False):
        # x shape: (batch_size, n_timestep, n_feature)
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # x shape after LSTM: (batch_size, n_hidden)
        output = self.out(x)
        output = tf.nn.softmax(output, axis=1)
        output = tf.map_fn(lambda w: self.rebalance(w, self.lb, self.ub), output)
        return output

    def rebalance(self, weight, lb, ub):
        old = weight
        weight_clamped = tf.clip_by_value(old, lb, ub)
        while True:
            leftover = tf.reduce_sum(old - weight_clamped)
            nominees = tf.boolean_mask(weight_clamped, weight_clamped != ub)
            gift = leftover * (nominees / tf.reduce_sum(nominees))
            weight_clamped = tf.where(weight_clamped != ub, weight_clamped + gift, weight_clamped)
            old = weight_clamped
            if tf.reduce_sum(tf.cast(weight_clamped > ub, tf.int32)) == 0:
                break
            else:
                weight_clamped = tf.clip_by_value(old, lb, ub)
        return weight_clamped