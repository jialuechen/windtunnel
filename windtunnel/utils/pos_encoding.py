import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,embed_dim,max_len=5000):
        super().__init__(); pe=torch.zeros(max_len,embed_dim)
        position=torch.arange(max_len).unsqueeze(1).float()
        div_term=torch.exp(torch.arange(0,embed_dim,2).float()*-(math.log(10000)/embed_dim))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]