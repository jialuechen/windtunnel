import torch
import torch.nn as nn
from deepfolio.models import FeatureExtractor, ParameterPredictor, QPLayer, DeepLearningLayer, MergeLayer
from deepfolio.optimizers import CustomOptimizer
from deepfolio.data import get_data_loader
from deepfolio.utils import sharpe_ratio, max_drawdown

class DiffOptPortfolio(nn.Module):
    def __init__(self, input_dim, n_assets, hidden_dim):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim)
        self.parameter_predictor = ParameterPredictor(hidden_dim, n_assets)
        self.qp_layer = QPLayer(n_assets)
        self.dl_layer = DeepLearningLayer(hidden_dim, hidden_dim, n_assets)
        self.merge_layer = MergeLayer(n_assets)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        mu, sigma = self.parameter_predictor(features)
        qp_output = self.qp_layer(mu, sigma)
        dl_output = self.dl_layer(features)
        return self.merge_layer(qp_output, dl_output)

def train(model, train_loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for features, returns in train_loader:
            optimizer.zero_grad()
            weights = model(features)
            portfolio_returns = torch.sum(weights * returns, dim=1)
            loss = -sharpe_ratio(portfolio_returns)  # Maximize Sharpe ratio
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def main():
    # Assume we have prepared our data
    features, returns = prepare_data()
    train_loader = get_data_loader(features, returns)
    
    input_dim = features.shape[1]
    n_assets = returns.shape[1]
    hidden_dim = 64
    
    model = DiffOptPortfolio(input_dim, n_assets, hidden_dim)
    optimizer = CustomOptimizer(model.parameters())
    
    train(model, train_loader, optimizer)

if __name__ == "__main__":
    main()