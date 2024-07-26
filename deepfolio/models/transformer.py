import tensorflow as tf
import numpy as np
import math

def create_mask(batch, sequence_length):
    mask = tf.zeros((batch, sequence_length, sequence_length))
    for i in range(sequence_length):
        mask = mask[:, i, :i+1].assign(1)
    return mask

class Norm(tf.keras.layers.Layer):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.eps = eps
        self.alpha = tf.Variable(tf.ones(self.size))
        self.bias = tf.Variable(tf.zeros(self.size))

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

def attention(q, k, v, d_k, mask=None, dropout=None, return_weights=False):
    scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))
    
    if mask is not None:
        scores += (mask * -1e9)
    
    scores = tf.nn.softmax(scores, axis=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = tf.matmul(scores, v)
    
    if return_weights:
        return output, scores
    return output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = tf.keras.layers.Dense(d_model)
        self.v_linear = tf.keras.layers.Dense(d_model)
        self.k_linear = tf.keras.layers.Dense(d_model)
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.out = tf.keras.layers.Dense(d_model)
    
    def call(self, q, k, v, mask=None, return_weights=False):
        bs = tf.shape(q)[0]
        
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        k = tf.reshape(k, (bs, -1, self.h, self.d_k))
        q = tf.reshape(q, (bs, -1, self.h, self.d_k))
        v = tf.reshape(v, (bs, -1, self.h, self.d_k))
        
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        if return_weights:
            scores, weights = attention(q, k, v, self.d_k, mask, self.dropout, return_weights=return_weights)
        else:
            scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        concat = tf.transpose(scores, perm=[0, 2, 1, 3])
        concat = tf.reshape(concat, (bs, -1, self.d_model))
        output = self.out(concat)
        
        if return_weights:
            return output, weights
        else:
            return output

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=400, dropout=0.1):
        super().__init__()
        self.linear_1 = tf.keras.layers.Dense(d_ff)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        x = self.dropout(tf.nn.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = tf.keras.layers.Dropout(dropout)
        self.dropout_2 = tf.keras.layers.Dropout(dropout)
    
    def call(self, x, mask=None, return_weights=False):
        x2 = self.norm_1(x)
        if return_weights:
            attn_output, attn_weights = self.attn(x2, x2, x2, mask, return_weights=return_weights)
        else:
            attn_output = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(attn_output)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        if return_weights:
            return x, attn_weights
        else:
            return x

class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        pe = np.zeros((max_seq_len, d_model))
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        seq_len = tf.shape(x)[1]
        pe = self.pe[:seq_len, :]
        return self.dropout(x + pe)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_size, seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(input_size, seq_len, dropout=dropout)
        self.layers = [EncoderLayer(input_size, heads, dropout) for _ in range(N)]
        self.norm = Norm(input_size)
    
    def call(self, x, mask=None, return_weights=False):
        x = self.pe(x)
        for i in range(self.N):
            if i == 0 and return_weights:
                x, weights = self.layers[i](x, mask=mask, return_weights=return_weights)
            else:
                x = self.layers[i](x, mask=mask)
        
        if return_weights:
            return self.norm(x), weights
        else:
            return self.norm(x)

class Transformer(tf.keras.Model):
    def __init__(self, n_feature, n_timestep, n_layer, n_head, n_dropout, n_output, lb, ub):
        super().__init__()
        self.encoder = Encoder(n_feature, n_timestep, n_layer, n_head, n_dropout)
        self.out = tf.keras.layers.Dense(n_output)
        self.tempmaxpool = tf.keras.layers.GlobalMaxPooling1D()
        self.lb = lb
        self.ub = ub
    
    def call(self, src, return_weights=False):
        mask = create_mask(tf.shape(src)[0], tf.shape(src)[1])
        
        if return_weights:
            e_outputs, weights = self.encoder(src, mask, return_weights=return_weights)
        else:
            e_outputs = self.encoder(src, mask)
        
        e_outputs = self.tempmaxpool(e_outputs)
        output = self.out(e_outputs)
        output = tf.nn.softmax(output, axis=1)
        output = tf.map_fn(lambda x: self.rebalance(x, self.lb, self.ub), output)
        
        if return_weights:
            return output, weights
        else:
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
