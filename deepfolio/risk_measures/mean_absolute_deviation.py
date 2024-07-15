class MeanAbsoluteDeviation(keras.layers.Layer):
    def call(self, inputs):
        mean_return = tf.reduce_mean(inputs, axis=1, keepdims=True)
        return tf.reduce_mean(tf.abs(inputs - mean_return), axis=1)