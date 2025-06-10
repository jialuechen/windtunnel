import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def get_torch_trans(d_model, nhead, num_layers, dropout=0.1):
    layer = TransformerEncoderLayer(
        d_model=d_model, nhead=nhead,
        dim_feedforward=d_model*4,
        dropout=dropout, activation='gelu'
    )
    return TransformerEncoder(layer, num_layers=num_layers)

def get_linear_trans(d_model, nhead, num_layers, dropout=0.1):
    # Simplified linear attention placeholder
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        ))
    return nn.Sequential(*layers)