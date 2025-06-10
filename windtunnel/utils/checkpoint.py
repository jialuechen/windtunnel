import torch

def save_checkpoint(model,optimizer,path):
    torch.save({'model':model.state_dict(),'optim':optimizer.state_dict()},path)

def load_checkpoint(model,optimizer,path,map_location=None):
    ckpt=torch.load(path,map_location=map_location)
    model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optim'])