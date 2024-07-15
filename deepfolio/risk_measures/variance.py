class Variance(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.reduce_variance(inputs, axis=1)