import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch.nn as nn

class QPLayer(nn.Module):
    def __init__(self, n_assets):
        super().__init__()
        mu = cp.Parameter(n_assets)
        Sigma = cp.Parameter((n_assets, n_assets))
        w = cp.Variable(n_assets)
        
        obj = cp.Minimize(cp.quad_form(w, Sigma) - mu.T @ w)
        constraints = [cp.sum(w) == 1, w >= 0]
        
        problem = cp.Problem(obj, constraints)
        self.qp_layer = CvxpyLayer(problem, parameters=[mu, Sigma], variables=[w])
    
    def forward(self, mu, Sigma):
        return self.qp_layer(mu, Sigma)[0]