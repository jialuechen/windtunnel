import keras
import tensorflow as tf

class VariationOfInformation(keras.layers.Layer):
    def call(self, inputs):
        def entropy(x):
            hist = tf.histogram_fixed_width(x, [tf.reduce_min(x), tf.reduce_max(x)], nbins=20)
            probs = hist / tf.reduce_sum(hist)
            return -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))

        def mutual_information(x, y):
            joint_hist = tf.histogram2d(x, y, bins=[20, 20])[0]
            joint_probs = joint_hist / tf.reduce_sum(joint_hist)
            marginal_x = tf.reduce_sum(joint_probs, axis=1)
            marginal_y = tf.reduce_sum(joint_probs, axis=0)
            mi = tf.reduce_sum(joint_probs * tf.math.log(joint_probs / (marginal_x[:, tf.newaxis] * marginal_y[tf.newaxis, :]) + 1e-10))
            return mi

        n_assets = tf.shape(inputs)[2]
        vi_matrix = tf.zeros((tf.shape(inputs)[0], n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                h_i = entropy(inputs[:, :, i])
                h_j = entropy(inputs[:, :, j])
                mi = mutual_information(inputs[:, :, i], inputs[:, :, j])
                vi = h_i + h_j - 2 * mi
                vi_matrix = tf.tensor_scatter_nd_update(
                    vi_matrix, 
                    [[k, i, j] for k in range(tf.shape(inputs)[0])] + [[k, j, i] for k in range(tf.shape(inputs)[0])], 
                    tf.tile(vi[tf.newaxis], [2 * tf.shape(inputs)[0]])
                )
        
        return vi_matrix