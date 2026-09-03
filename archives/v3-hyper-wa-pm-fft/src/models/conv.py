# src/models/conv.py

import torch
import torch.nn as nn

class FFT_Field_Predictor(nn.Module):
    """
    A neural network for predicting fields in the Fourier domain.
    """
    def __init__(self, in_ch=12, hidden=64, out_ch=12, kernel_size=5):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden = hidden
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden, out_ch, kernel_size, padding=padding)
        )
        

    def forward(self, x):
        return self.net(x)

    def save_model(self, filepath):
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
