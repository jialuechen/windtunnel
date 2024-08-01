import torch.nn as nn

class ParameterPredictor(nn.Module):
    def __init__(self, input_dim, n_assets):
        super().__init__()
        self.mu_predictor = nn.Linear(input_dim, n_assets)
        self.sigma_predictor = nn.Linear(input_dim, n_assets * n_assets)
    
    def forward(self, x):
        mu = self.mu_predictor(x)
        sigma = self.sigma_predictor(x).view(-1, mu.size(1), mu.size(1))
        return mu, sigma