"""Surrogate forward model: (wl, ang, amp, per) -> power efficiencies."""

import torch
import torch.nn as nn


class SurrogateMLP(nn.Module):
    """Multi-layer perceptron surrogate for the sine TiN optical response."""

    def __init__(self, in_dim=4, hidden=128, out_dim=4, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SurrogateEnsemble(nn.Module):
    """Average of ``n_members`` independent surrogates (lower variance)."""

    def __init__(self, n_members=3, **kwargs):
        super().__init__()
        self.members = nn.ModuleList(
            [SurrogateMLP(**kwargs) for _ in range(n_members)]
        )

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], dim=0).mean(dim=0)