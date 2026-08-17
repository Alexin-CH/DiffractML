# src/models/linear.py

import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, in_ch=2, hidden=64, out_ch=128):
        super(LinearModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_ch, 2**4), nn.Tanh(),
            nn.Linear(2**4, 2**6), nn.Tanh(),
            nn.Linear(2**6, 2**8), nn.Tanh(),
            nn.Linear(2**8, out_ch)
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
