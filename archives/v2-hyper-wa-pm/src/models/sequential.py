# src/models/sequential.py

import torch
import torch.nn as nn

class SequentialModel(nn.Module):
    def __init__(self, model_a, model_b):
        super(SequentialModel, self).__init__()
        self.nn = nn.Sequential(
            model_a,
            model_b
        )
    def forward(self, x):
        """
        Forward pass through the sequential model.
        :param x: Input to the parent model.
        :param y: Input to the child model.
        :return: Output of the model.
        """
        return self.nn(x)

    def save_model(self, filepath):
        """Save the trained model to the specified filepath."""
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """Load a pre-trained model from the specified filepath."""
        self.load_state_dict(torch.load(filepath))
        self.eval()
        return self
