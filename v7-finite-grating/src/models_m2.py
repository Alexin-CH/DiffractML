"""M2 correction network: geometry + fast-RCWA spectrum -> spectral residual."""

from __future__ import annotations

import torch
import torch.nn as nn


class CorrectionMLP(nn.Module):
    """Predict ``[dR, dT]`` to add to the fast RCWA ``(R, T)``."""

    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 4,
                 out_dim: int = 2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
