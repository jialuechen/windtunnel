import numpy as np

def dtw_distance(x, y):
    n, m = len(x), len(y)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - y[j-1])
            dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    return dtw[n,m]