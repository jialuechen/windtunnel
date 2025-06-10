import torch
from windtunnel.diffusion.csdi import DiffusionEmbedding, ResidualBlock, CSDIModel

def test_embedding_and_resblock():
    emb_layer = DiffusionEmbedding(dim=8)
    t = torch.randint(0, 100, (4,))
    emb = emb_layer(t)
    assert emb.shape == (4, 8)
    rb = ResidualBlock(dim=8)
    x = torch.randn(2, 10, 8)
    out = rb(x)
    assert out.shape == x.shape

def test_csdi_model():
    model = CSDIModel(input_dim=1, dim=8, layers=2)
    x = torch.randn(3, 15, 8)
    t = torch.randint(0, 100, (3,))
    out = model(x, t)
    assert out.shape == (3, 15, 1)