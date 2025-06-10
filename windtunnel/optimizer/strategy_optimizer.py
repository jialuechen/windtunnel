import pandas as pd
import random
from abc import ABC, abstractmethod
from sklearn.model_selection import ParameterGrid
from windtunnel.api import Simulator

class Sampler(ABC):
    @abstractmethod
    def sample(self, param_grid): pass

class GridSampler(Sampler):
    def sample(self,param_grid): yield from ParameterGrid(param_grid)

class RandomSampler(Sampler):
    def __init__(self,n_iter=10,random_state=None):
        self.n_iter=n_iter
        if random_state: random.seed(random_state)
    def sample(self,param_grid):
        keys=list(param_grid)
        for _ in range(self.n_iter): yield {k: random.choice(param_grid[k]) for k in keys}

class AutoStrategyOptimizer:
    def __init__(self,simulator_config,evaluate_fn,filter_rules=None,sampler=None):
        self.sim_config=simulator_config; self.evaluate_fn=evaluate_fn
        self.filter_rules=filter_rules or {}; self.sampler=sampler or GridSampler()
    def optimize(self,param_grid):
        records=[]
        for params in self.sampler.sample(param_grid):
            sim=Simulator({**self.sim_config,'diffusion':params})
            res=self.evaluate_fn(sim,params)
            rec={**params,**(res if isinstance(res,dict) else {'metric':float(res)})}
            records.append(rec)
        df=pd.DataFrame(records)
        if 'metric' in df: df=df.sort_values('metric',ascending=False)
        return df.reset_index(drop=True)
    def filter(self,df):
        for m,thr in self.filter_rules.items():
            if m in df: df=df[df[m]>=thr]
        return df.reset_index(drop=True)