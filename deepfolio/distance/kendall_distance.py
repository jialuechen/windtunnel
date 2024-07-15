import keras
import tensorflow as tf

class KendallDistance(keras.layers.Layer):
    def call(self, inputs):
        def kendall_tau(x, y):
            n = tf.shape(x)[0]
            pairs = tf.shape(x)[0] * (tf.shape(x)[0] - 1) // 2
            concordant = tf.reduce_sum(tf.sign(x[None, :] - x[:, None]) * tf.sign(y[None, :] - y[:, None]))
            tau = concordant / tf.cast(pairs, tf.float32)
            return 1.0 - tau

        n_assets = tf.shape(inputs)[2]
        distance_matrix = tf.zeros((tf.shape(inputs)[0], n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                distance = kendall_tau(inputs[:, :, i], inputs[:, :, j])
                distance_matrix = tf.tensor_scatter_nd_update(
                    distance_matrix, 
                    [[k, i, j] for k in range(tf.shape(inputs)[0])] + [[k, j, i] for k in range(tf.shape(inputs)[0])], 
                    tf.tile(distance[tf.newaxis], [2 * tf.shape(inputs)[0]])
                )
        
        return distance_matrix