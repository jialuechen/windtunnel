import torch
import torch.nn as nn
from .diff_models import get_torch_trans
from windtunnel.utils.pos_encoding import PositionalEncoding

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.transformer = get_torch_trans(
            d_model=embed_dim, nhead=num_heads,
            num_layers=num_layers, dropout=dropout
        )
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = x.transpose(0,1)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0,1)
        return self.output_proj(x)