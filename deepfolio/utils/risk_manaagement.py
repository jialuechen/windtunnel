import tensorflow as tf

def value_at_risk(returns, confidence_level=0.95):
    return tf.quantile(returns, 1 - confidence_level)

def conditional_value_at_risk(returns, confidence_level=0.95):
    var = value_at_risk(returns, confidence_level)
    return tf.reduce_mean(tf.boolean_mask(returns, returns <= var))

def tracking_error(portfolio_returns, benchmark_returns):
    return tf.math.reduce_std(portfolio_returns - benchmark_returns)

class RiskManager(tf.keras.layers.Layer):
    def __init__(self, max_leverage=1.5, max_position_size=0.3):
        super(RiskManager, self).__init__()
        self.max_leverage = max_leverage
        self.max_position_size = max_position_size
    
    def call(self, weights):
        weights = tf.maximum(weights, 0)
        weights = tf.minimum(weights, self.max_position_size)
        weights = weights / tf.reduce_sum(weights)
        
        leverage = tf.reduce_sum(tf.abs(weights))
        if leverage > self.max_leverage:
            weights = weights * (self.max_leverage / leverage)
        
        return weights