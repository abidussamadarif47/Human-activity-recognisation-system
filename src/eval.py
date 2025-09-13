import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from src.data import UCIHARDataset
from src.models import CNN1D

def evaluate(model_path, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = UCIHARDataset(data_dir, split='test')
    loader = torch.utils.data.DataLoader(ds, batch_size=128)
    model = CNN1D(in_channels=ds.X.shape[2], n_classes=6)
    model.load_state_dict(torch.load(model_path)['model_state'])
    model.to(device)
    model.eval()
    ys = []
    ypred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            ys.extend(y.numpy().tolist())
            ypred.extend(logits.argmax(1).cpu().numpy().tolist())
    print(classification_report(ys, ypred))
    cm = confusion_matrix(ys, ypred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Example usage:
# evaluate('./experiments/best.pth', './data')# Evaluation code here
