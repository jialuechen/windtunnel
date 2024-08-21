import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Embedding, Input
from tensorflow.keras.models import Model
import numpy as np

# Time2Vec layer
class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, kernel_size):
        super(Time2Vec, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.weights = [
            self.add_weight(name='weight', shape=(input_shape[-1], 1), initializer='uniform', trainable=True),
            self.add_weight(name='bias', shape=(input_shape[-1],), initializer='uniform', trainable=True)
        ]
        self.built = True

    def call(self, x):
        time_features = tf.math.sin(tf.matmul(x, self.weights[0]) + self.weights[1])
        return tf.concat([x, time_features], -1)

# Gated Residual Network (GRN)
class GatedResidualNetwork(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GatedResidualNetwork, self).__init__()
        self.dense_1 = Dense(units, activation='elu')
        self.dense_2 = Dense(units, activation=None)
        self.gate = Dense(units, activation='sigmoid')
        self.norm = LayerNormalization()

    def call(self, x):
        output = self.dense_1(x)
        output = self.dense_2(output)
        gated_output = self.gate(output)
        output = gated_output * output
        return self.norm(x + output)

# Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.grn = GatedResidualNetwork(dff)
        self.norm = LayerNormalization()

    def call(self, x):
        attn_output = self.mha(x, x, x)
        out1 = self.norm(x + attn_output)
        grn_output = self.grn(out1)
        return grn_output

# Decoder layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.grn = GatedResidualNetwork(dff)
        self.norm = LayerNormalization()

    def call(self, x, enc_output):
        attn1 = self.mha1(x, x, x)
        out1 = self.norm(x + attn1)
        attn2 = self.mha2(out1, enc_output, enc_output)
        out2 = self.norm(out1 + attn2)
        grn_output = self.grn(out2)
        return grn_output

# Portfolio Transformer model
def portfolio_transformer(input_shape, d_model, num_heads, dff, num_layers):
    inputs = Input(shape=input_shape)

    # Time2Vec embedding
    time_embedding = Time2Vec(input_shape[-1])(inputs)

    # Encoder
    enc_output = time_embedding
    for _ in range(num_layers):
        enc_output = EncoderLayer(d_model, num_heads, dff)(enc_output)

    # Decoder
    dec_output = enc_output
    for _ in range(num_layers):
        dec_output = DecoderLayer(d_model, num_heads, dff)(dec_output, enc_output)

    # Output layer with short-selling support
    final_output = Dense(input_shape[-1])(dec_output)
    portfolio_weights = tf.nn.softmax(final_output)

    # Model
    model = Model(inputs=inputs, outputs=portfolio_weights)
    return model

# Example usage
input_shape = (30, 10)  # 30 time steps, 10 assets
d_model = 64
num_heads = 4
dff = 128
num_layers = 4

model = build_portfolio_transformer(input_shape, d_model, num_heads, dff, num_layers)
model.compile(optimizer='adam', loss='mse')  # Adjust loss as per the paper's requirements
model.summary()