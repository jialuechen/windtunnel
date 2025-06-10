import torch
import numpy as np
from windtunnel.diffusion.ddpm import DiffusionEngine

def test_forward_and_sample():
    engine = DiffusionEngine(input_dim=1, timesteps=10)
    x0 = torch.randn(2, 10, 1)
    cond = torch.randn(2, 1, 5, 1)
    mask = torch.ones(2, 10, dtype=torch.bool)
    xt, noise = engine.forward_diffusion(x0, t=5)
    assert xt.shape == x0.shape and noise.shape == x0.shape
    out = engine.sample(cond, mask, length=10)
    assert out.shape == (2, 10, 1)