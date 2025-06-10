import numpy as np

def pearson_similarity(x, y):
    if x.ndim!=1 or y.ndim!=1:
        raise ValueError("Inputs must be 1D arrays")
    if len(x)!=len(y):
        raise ValueError("Inputs must have same length")
    xm, ym = x.mean(), y.mean()
    num = np.sum((x-xm)*(y-ym))
    den = np.sqrt(np.sum((x-xm)**2)*np.sum((y-ym)**2))
    return num/den if den!=0 else 0.0