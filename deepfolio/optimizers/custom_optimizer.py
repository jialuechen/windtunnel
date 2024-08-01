import tensorflow as tf

class CustomOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, name="CustomOptimizer", **kwargs):
        super(CustomOptimizer, self).__init__(name, **kwargs)
        self._lr = learning_rate
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum")
    
    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state):
        var_dtype = var.dtype.base_dtype
        lr_t = self._lr
        momentum = self.get_slot(var, "momentum")
        
        momentum_t = momentum.assign(0.9 * momentum + 0.1 * grad)
        var_update = var.assign_sub(lr_t * momentum_t)
        
        return tf.group(*[var_update, momentum_t])
    
    def get_config(self):
        base_config = super(CustomOptimizer, self).get_config()
        return {**base_config, "learning_rate": self._lr}