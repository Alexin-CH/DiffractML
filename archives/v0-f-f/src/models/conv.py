# src/conv.py

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FieldPredictorV0(nn.Module):
    def __init__(self, in_ch=12, hidden=64, out_ch=12):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()
        return self

# end of file
