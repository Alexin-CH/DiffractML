"""Experiment 2 models: shared geometry encoder + residual head.

The encoder trunk is wavelength-independent: it produces a latent ``z_geo``
(which carries defect information to the correction head) and a small config
delta used to run the differentiable RCWA.  Given the gradient conditioning
found by the probe, the config head predicts an **effective amplitude** and a
**small effective-period delta**; base is constant and padding/sharpness are
fixed (Experiment 1 showed they are unnecessary).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeometryEncoder(nn.Module):
    """``descriptors -> z_geo (latent), cfg_delta``."""

    def __init__(self, in_dim: int = 8, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent),
        )
        self.cfg_head = nn.Linear(latent, 2)  # [dlog_amp, dlog_period]
        self.latent = latent

    def forward(self, d):
        z = self.trunk(d)
        cfg = torch.tanh(self.cfg_head(z))  # bounded to [-1, 1]
        return z, cfg


class ResidualHead(nn.Module):
    """``[z_geo, rcwa_R, rcwa_T, rcwa_A, wavelength, descriptors] -> [dR, dT]``."""

    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 3, out_dim: int = 2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
