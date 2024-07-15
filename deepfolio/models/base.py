import keras

class BaseModel(keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        raise NotImplementedError("Subclasses must implement this method")

    def fit(self, returns, **kwargs):
        super().fit(returns, returns, **kwargs)

    def predict(self, returns):
        return super().predict(returns)



