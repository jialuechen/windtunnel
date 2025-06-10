import torch

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total=0
    for cond,target,mask in dataloader:
        cond, target, mask = cond.to(device), target.to(device), mask.to(device)
        loss = model.training_step(target,cond,mask)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total+=loss.item()
    return total/len(dataloader)