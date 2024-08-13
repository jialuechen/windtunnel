import tensorflow as tf
import numpy as np
import scipy.cluster.hierarchy as sch

class HierarchicalRiskParityOptimizer(tf.keras.layers.Layer):
    def __init__(self, n_assets):
        super(HierarchicalRiskParityOptimizer, self).__init__()
        self.n_assets = n_assets

    def call(self, inputs):
        returns, = inputs
        
        def hrp_optimization(returns):
            # Calculate correlation matrix
            corr = np.corrcoef(returns.T)
            
            # Distance matrix
            dist = np.sqrt(0.5 * (1 - corr))
            
            # Hierarchical clustering
            link = sch.linkage(dist, 'single')
            sortIx = sch.leaves_list(link)
            
            # Sort correlation matrix
            corr = corr[sortIx, :][:, sortIx]
            
            # Recursive bisection
            weights = np.ones(self.n_assets)
            clusters = [list(range(self.n_assets))]
            while len(clusters) > 0:
                clusters = [cl[start:end] for cl in clusters
                            for start, end in ((0, len(cl) // 2), (len(cl) // 2, len(cl)))
                            if len(cl) > 1]
                for i in range(0, len(clusters), 2):
                    cl1 = clusters[i]
                    cl2 = clusters[i + 1]
                    var1 = np.sum(np.var(returns[:, cl1], axis=0))
                    var2 = np.sum(np.var(returns[:, cl2], axis=0))
                    alpha = 1 - var1 / (var1 + var2)
                    weights[cl1] *= alpha
                    weights[cl2] *= 1 - alpha
            
            # Revert to original order
            weights = weights[np.argsort(sortIx)]
            return weights / np.sum(weights)

        optimized_w = tf.py_function(
            func=hrp_optimization,
            inp=[returns],
            Tout=tf.float32
        )
        
        return optimized_w

class HRPDiffOptPortfolio(tf.keras.Model):
    def __init__(self, input_dim, n_assets, hidden_dim):
        super(HRPDiffOptPortfolio, self).__init__()
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        self.returns_predictor = tf.keras.layers.Dense(n_assets)
        self.hrp_optimizer = HierarchicalRiskParityOptimizer(n_assets)
    
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        returns = self.returns_predictor(features)
        return self.hrp_optimizer([returns])