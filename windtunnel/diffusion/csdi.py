import torch
import torch.nn as nn
import math

class DiffusionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
    def forward(self, t):
        half = t.shape[1] // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device)*-emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.proj(emb)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, dim), nn.GELU(), nn.Conv1d(dim, dim, 3, padding=1)
        )
    def forward(self, x):
        return x + self.net(x.transpose(1,2)).transpose(1,2)

class CSDIModel(nn.Module):
    def __init__(self, input_dim, dim, layers):
        super().__init__()
        self.embed = DiffusionEmbedding(dim)
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(layers)])
        self.out = nn.Conv1d(dim, input_dim, 1)
    def forward(self, x, t):
        emb = self.embed(t)
        h = x + emb.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h.transpose(1,2)).transpose(1,2)