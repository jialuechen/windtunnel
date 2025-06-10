from torch.optim.lr_scheduler import CosineAnnealingLR

def get_scheduler(optimizer,T_max,eta_min=0):
    return CosineAnnealingLR(optimizer,T_max=T_max,eta_min=eta_min)