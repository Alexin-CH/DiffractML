# src/models/linear.py

import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, in_ch=2, hidden=64, out_ch=128):
        super(LinearModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_ch)
        )

    def forward(self, x):
        return self.nn(x)

    def save_model(self, filepath):
        """Save the trained model to the specified filepath."""
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """Load a pre-trained model from the specified filepath."""
        self.load_state_dict(torch.load(filepath))
        self.eval()
        return self

class OneLayer(nn.Module):
    def __init__(self, in_ch=2, out_ch=128):
        super(OneLayer, self).__init__()
        self.nn = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        x = x.view(x.size(1), -1)
        y = self.nn(x)
        return y

    def save_model(self, filepath):
        """Save the trained model to the specified filepath."""
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """Load a pre-trained model from the specified filepath."""
        self.load_state_dict(torch.load(filepath))
        self.eval()
        return self

# end of file
