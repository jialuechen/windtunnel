import torch
import torch.nn.functional as F
import numpy as np
from .transformer import TimeSeriesTransformer

class DiffusionEngine(torch.nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 8,
                 dropout: float = 0.1, timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.model = TimeSeriesTransformer(
            input_dim=input_dim, embed_dim=embed_dim,
            num_layers=num_layers, num_heads=num_heads,
            dropout=dropout
        )

    def forward_diffusion(self, x0: torch.Tensor, t: int):
        sqrt_ab = torch.sqrt(self.alpha_bar[t])
        sqrt_1_ab = torch.sqrt(1 - self.alpha_bar[t])
        noise = torch.randn_like(x0)
        return sqrt_ab * x0 + sqrt_1_ab * noise, noise

    def denoise_step(self, xt, cond, mask, t):
        batch, _, d = xt.shape[0], xt.shape[1], xt.shape[2]
        K, Th, _ = cond.shape[1], cond.shape[2], cond.shape[3]
        cond_flat = cond.view(batch, K*Th, d)
        inp = torch.cat([cond_flat, xt], dim=1)
        pad_mask = torch.cat([
            torch.ones(batch, K*Th, dtype=torch.bool, device=xt.device),
            ~mask
        ], dim=1)
        eps_pred = self.model(inp, src_key_padding_mask=pad_mask)
        eps_pred = eps_pred[:, -xt.size(1):, :]
        beta_t, alpha_t, alpha_bar_t = self.beta[t], self.alpha[t], self.alpha_bar[t]
        coef1 = 1/alpha_t.sqrt()
        coef2 = beta_t/torch.sqrt(1-alpha_bar_t)
        return coef1*(xt - coef2*eps_pred)

    def sample(self, cond, mask, length):
        batch, _, _, d = cond.shape
        xt = torch.randn(batch, length, d, device=next(self.parameters()).device)
        for t in reversed(range(self.timesteps)):
            xt = self.denoise_step(xt, cond, mask, t)
        return xt

    def training_step(self, x0, cond, mask):
        t = np.random.randint(0, self.timesteps)
        xt, noise = self.forward_diffusion(x0, t)
        denoised = self.denoise_step(xt, cond, mask, t)
        return F.mse_loss(denoised, x0)