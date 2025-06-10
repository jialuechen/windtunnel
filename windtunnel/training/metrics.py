import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(((y_true-y_pred)**2).mean())

def mae(y_true, y_pred):
    return np.abs(y_true-y_pred).mean()