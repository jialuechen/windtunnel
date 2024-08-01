import torch

def sharpe_ratio(returns, risk_free_rate=0):
    return (torch.mean(returns) - risk_free_rate) / torch.std(returns)

def max_drawdown(returns):
    cumulative = torch.cumsum(returns, dim=0)
    max_so_far = torch.maximum.accumulate(cumulative)
    drawdowns = (max_so_far - cumulative) / max_so_far
    return torch.max(drawdowns)