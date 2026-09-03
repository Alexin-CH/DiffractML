# src/models/linear.py

import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, in_ch=2, hidden=64, out_ch=128):
        super(LinearModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_ch, 8), nn.Tanh(),
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 32), nn.Tanhshrink(),
            nn.Linear(32, 64)
        )

    def forward(self, x):
        return self.nn(x)

    def save_model(self, filepath):
        """Save the trained model to the specified filepath."""
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath, mode='eval'):
        self.load_state_dict(torch.load(filepath))
        if mode == 'eval':
            self.eval()
        elif mode == 'train':
            self.train()
        else:
            raise ValueError("Mode must be 'eval' or 'train'.")
        return self

# end of file
