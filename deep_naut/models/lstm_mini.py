import torch
import torch.nn as nn

class MiniLSTM(nn.Module):
    def __init__(self, f=8, h=64, l=2, ncls=2):
        super().__init__()
        self.lstm = nn.LSTM(f, h, num_layers=l, batch_first=True)
        self.fc = nn.Linear(h, ncls)

    def forward(self, x):  # x: (B,T,F)
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])  # ロジット（2クラスなら shape=(B,2)）
