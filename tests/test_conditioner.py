import numpy as np
import torch
from windtunnel.conditioning.conditioner import build_condition_input

def test_build_condition_input():
    cond = np.arange(12).reshape(1, 2, 3, 2).astype(float)
    target = np.zeros((1, 4, 2))
    mask = np.array([[False, True, True, False]])
    inp, mask_tensor = build_condition_input(cond, target, mask)
    assert inp.shape == (1, 2*3 + 4, 2)
    assert mask_tensor.shape == (1, 2*3 + 4)