"""Experiment 3 models: geometry encoders and the config head.

Two encoders are provided:

* :class:`GlobalEncoder` — the Experiment 2 design: eight hand-made global
  descriptors (period, amplitude, base, ridge-scale mean/std/min, missing
  count, missing index).
* :class:`RidgeSetEncoder` — the redesign: a permutation-aware DeepSets
  encoder over one token per ridge (height scale, missing flag, normalised
  position), pooled and combined with five global scalars.  The global
  ``mean/std/min`` statistics throw away *which* ridge is defective; the token
  set keeps it.

:class:`ConfigHead` predicts an effective geometry for the differentiable
RCWA.  ``amp`` uses a bounded log-amplitude delta; ``amp_period`` adds a
log-period delta, deliberately bounded to ±3% because the RCWA period gradient
was noisy and sign-inconsistent at the Experiment 2 operating point (see
``reports/experiments/E2-differentiable-encoder.md``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

AMP_SCALE = 0.15      # exp(+-0.15) ~ +-16% effective amplitude
PERIOD_SCALE = 0.03   # exp(+-0.03) ~ +-3% effective period


class GlobalEncoder(nn.Module):
    """``descriptors (8) -> z_geo (latent)``."""

    def __init__(self, in_dim: int = 8, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent),
        )
        self.latent = latent
        self.out_dim = latent

    def forward(self, d):
        return self.net(d)


class RidgeSetEncoder(nn.Module):
    """``(ridge tokens, global scalars) -> z_geo``.

    ``tokens`` has shape ``(n_ridges, 3)``: height scale, missing flag and
    normalised ridge position.  ``globals`` has shape ``(5,)``: scaled period,
    amplitude, base, missing count and normalised missing index.
    """

    def __init__(self, token_dim: int = 3, glob_dim: int = 5,
                 hidden: int = 64, latent: int = 16):
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + glob_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, latent),
        )
        self.latent = latent
        self.out_dim = latent

    def forward(self, tokens, glob):
        h = self.token(tokens)                       # (n, ridges, hidden)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        return self.head(torch.cat([pooled, glob], dim=-1))


class ConfigHead(nn.Module):
    """``z_geo -> (log_amp, log_period)`` bounded by the scales above.

    ``mode="none"`` returns ``(None, None)`` so the ablation and the redesign
    share one code path.
    """

    def __init__(self, latent: int = 16, mode: str = "amp"):
        super().__init__()
        if mode not in ("none", "amp", "amp_period"):
            raise ValueError(f"unknown config mode {mode!r}")
        self.mode = mode
        out_dim = {"none": 0, "amp": 1, "amp_period": 2}[mode]
        self.fc = nn.Linear(latent, out_dim) if out_dim else None

    def forward(self, z):
        if self.fc is None:
            return None, None
        raw = torch.tanh(self.fc(z))                 # (n, 1 or 2)
        log_amp = AMP_SCALE * raw[:, 0]
        log_period = PERIOD_SCALE * raw[:, 1] if raw.shape[1] > 1 else None
        return log_amp, log_period
