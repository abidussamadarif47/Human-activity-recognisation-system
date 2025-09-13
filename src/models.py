import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, in_channels=9, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        h = self.net(x)
        h = h.squeeze(-1)
        out = self.fc(h)
        return out

class CNN_LSTM(nn.Module):
    def __init__(self, in_channels=9, n_classes=6, hidden=128, n_layers=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden*2, n_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        h = self.cnn(x) # (batch, feat, seq')
        h = h.permute(0,2,1) # (batch, seq', feat)
        out, _ = self.lstm(h)
        out = out.mean(dim=1)
        return self.classifier(out)
