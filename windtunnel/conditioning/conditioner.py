import numpy as np
import torch

def build_condition_input(cond, target, mask):
    batch, K, Th, d = cond.shape
    T = target.shape[1]
    cond_flat = cond.reshape(batch, K*Th, d)
    inp = np.concatenate([cond_flat, target], axis=1)
    mask_prefix = np.zeros((batch, K*Th), dtype=bool)
    mask_full = np.concatenate([mask_prefix, mask], axis=1)
    return torch.from_numpy(inp).float(), torch.from_numpy(mask_full)