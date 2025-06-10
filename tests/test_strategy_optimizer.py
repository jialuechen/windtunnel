import pandas as pd
from windtunnel.optimizer.strategy_optimizer import (
    AutoStrategyOptimizer, GridSampler, RandomSampler
)
from windtunnel.api import Simulator

def dummy_eval(sim: Simulator, params: dict):
    result = sim.generate(symbol="000300.SH", frequency="1d", length=5)
    arr = result.cpu().numpy()
    return {'mean': float(arr.mean()), 'std': float(arr.std())}

def test_optimize_grid():
    optimizer = AutoStrategyOptimizer(
        simulator_config={'diffusion': {'input_dim': 1, 'timesteps': 5}},
        evaluate_fn=dummy_eval,
        filter_rules=None,
        sampler=GridSampler()
    )
    grid = {'timesteps': [3, 5]}
    df = optimizer.optimize(grid)
    assert set(df['timesteps']) == {3, 5}
    assert 'mean' in df.columns and 'std' in df.columns

def test_optimize_random_and_filter():
    optimizer = AutoStrategyOptimizer(
        simulator_config={'diffusion': {'input_dim': 1, 'timesteps': 5}},
        evaluate_fn=dummy_eval,
        filter_rules={'mean': 0.0},
        sampler=RandomSampler(n_iter=2, random_state=0)
    )
    grid = {'timesteps': [3, 5]}
    df = optimizer.optimize(grid)
    df_f = optimizer.filter(df)
    assert all(df_f['mean'] >= 0.0)
    assert len(df) == 2