import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class UCIHARDataset(Dataset):
    def __init__(self, data_dir, split='train', use_raw_signals=True, transform=None):
        self.split = split
        self.transform = transform
        self.use_raw = use_raw_signals
        base = os.path.join(data_dir, split)
        # if aggregated features present
        X_path = os.path.join(data_dir, f'X_{split}.npy')
        y_path = os.path.join(data_dir, f'y_{split}.npy')
        if os.path.exists(X_path) and os.path.exists(y_path):
            self.X = np.load(X_path)
            self.y = np.load(y_path)
        else:
            # fallback to original UCI structure
            if use_raw_signals:
                signals = []
                signal_dir = os.path.join(data_dir, split, 'Inertial Signals')
                files = sorted([f for f in os.listdir(signal_dir) if f.endswith('.txt')])
                for f in files:
                    data = np.loadtxt(os.path.join(signal_dir, f))
                    # shape (n_samples, 128)
                    signals.append(data[..., np.newaxis])
                # concatenate along last dim -> (n_samples, 128, n_signals)
                X = np.concatenate(signals, axis=2)
            else:
                X = np.loadtxt(os.path.join(data_dir, f'X_{split}.txt'))
            # reshape if needed
            y = np.loadtxt(os.path.join(data_dir, f'y_{split}.txt')).astype(int) - 1
            self.X = X
            self.y = y
        # standardize per-channel
        n_samples, seq_len, n_ch = self.X.shape
        self.scaler = StandardScaler()
        X_reshaped = self.X.reshape(-1, n_ch)
        X_scaled = self.scaler.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_ch)
        self.X = X_scaled.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx] # (seq_len, channels)
        y = int(self.y[idx])
        if self.transform:
            x = self.transform(x)
        # convert to channels-first for PyTorch: (channels, seq_len)
        x = torch.from_numpy(x.transpose(1,0)).float()
        return x, y

def get_loaders(data_dir, batch_size=64, num_workers=4):
    train_ds = UCIHARDataset(data_dir, split='train')
    test_ds = UCIHARDataset(data_dir, split='test')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader