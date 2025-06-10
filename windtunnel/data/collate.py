import numpy as np

def collate_fn(batch):
    cond = np.stack([b[0] for b in batch])
    target = np.stack([b[1] for b in batch])
    mask = np.stack([b[2] for b in batch])
    return cond, target, mask