class SemiVariance(keras.layers.Layer):
    def call(self, inputs):
        mean_return = tf.reduce_mean(inputs, axis=1, keepdims=True)
        negative_returns = tf.minimum(inputs - mean_return, 0)
        return tf.reduce_mean(tf.square(negative_returns), axis=1)