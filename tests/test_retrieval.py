import numpy as np
from windtunnel.retrieval.dtw import dtw_distance
from windtunnel.retrieval.pearson import pearson_similarity

def test_dtw_distance():
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])
    assert dtw_distance(x, y) == 0.0

def test_pearson_similarity():
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])
    sim = pearson_similarity(x, y)
    assert abs(sim - 1.0) < 1e-6