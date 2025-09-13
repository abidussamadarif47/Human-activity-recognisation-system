import torch
import os

def save_checkpoint(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, path)

def load_checkpoint(model, path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    return model

def accuracy(preds, labels):
    return (preds.argmax(1) == labels).float().mean().item()
