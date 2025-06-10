import torch
from torch.utils.data import DataLoader,Dataset
from windtunnel.data.loader import load_dataset,process_data
from windtunnel.data.collate import collate_fn
from windtunnel.training.trainer import train_epoch
from windtunnel.training.metrics import rmse,mae
from windtunnel.utils.logger import get_logger
from windtunnel.utils.scheduler import get_scheduler
from windtunnel.utils.checkpoint import save_checkpoint
from windtunnel.api import Simulator

logger=get_logger()
class MarketDataset(Dataset):
    def __init__(self,seqs): self.data=seqs
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx]

def main():
    cfg=__import__('yaml').safe_load(open('config.yaml'))
    df=load_dataset(cfg['data']['path'])
    seqs=process_data(df,cfg['history_len'],cfg['pred_len'])
    loader=DataLoader(MarketDataset(seqs),batch_size=cfg['batch_size'],shuffle=True,collate_fn=collate_fn)
    sim=Simulator({'diffusion':{'input_dim':df.shape[1]-2}})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sim.engine.to(device)
    opt=torch.optim.AdamW(sim.engine.parameters(),lr=cfg['learning_rate'])
    sch=get_scheduler(opt,cfg['epochs'])
    for e in range(1,cfg['epochs']+1):
        loss=train_epoch(sim.engine,loader,opt,device)
        sch.step()
        logger.info(f"Epoch {e} loss={loss:.4f}")
        save_checkpoint(sim.engine,opt,f"checkpoints/epoch_{e}.pt")

if __name__=='__main__': main()