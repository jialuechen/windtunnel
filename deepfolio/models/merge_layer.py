import torch.nn as nn

class MergeLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(input_dim))
    
    def forward(self, qp_output, dl_output):
        w = torch.sigmoid(self.weight)
        return w * qp_output + (1 - w) * dl_output