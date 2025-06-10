import yaml,sys
from windtunnel.optimizer.strategy_optimizer import AutoStrategyOptimizer,RandomSampler
from windtunnel.api import Simulator

def evaluate(sim,params):
    data=sim.generate("000300.SH","1d",length=20)
    return float(data.mean().item())

if __name__=='__main__':
    cfg=yaml.safe_load(open('config.yaml'))
    optimizer=AutoStrategyOptimizer({'diffusion':{'input_dim':1}},evaluate,filter_rules=None,sampler=RandomSampler(5))
    grid={'timesteps':[10,20]}
    results=optimizer.optimize(grid)
    print(results)