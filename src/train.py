import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_loaders
from models import CNN1D

def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def train_loop(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders(cfg['data_dir'], batch_size=cfg['batch_size'])
    model = CNN1D(in_channels=cfg['in_channels'], n_classes=cfg['n_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    writer = SummaryWriter(log_dir=cfg['log_dir'])

    best_acc = 0.0
    for epoch in range(cfg['epochs']):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # eval
        model.eval()
        acc = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                acc += (logits.argmax(1) == y).sum().item()
        acc = acc / len(test_loader.dataset)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
        print(f'Epoch {epoch}: loss={epoch_loss:.4f} test_acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, optimizer, cfg['checkpoint_path'])
    writer.close()

if __name__ == '__main__':
    cfg = {
        'data_dir': './data',
        'batch_size': 64,
        'epochs': 30,
        'lr': 1e-3,
        'in_channels': 9,
        'n_classes': 6,
        'log_dir': './experiments/logs',
        'checkpoint_path': './experiments/best.pth'
    }
    train_loop(cfg)
