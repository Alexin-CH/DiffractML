"""PyTorch Dataset for the v6 generated CSV."""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset

PARAM_COLS = ["wl", "ang", "amp", "per"]
EFF_COLS = ["R_xx", "T_xx", "R_yy", "T_yy"]


class SinTinDataset(Dataset):
    """Maps (wl, ang, amp, per) -> power efficiencies [R_xx, T_xx, R_yy, T_yy].

    Inputs are z-scored using statistics collected from the training data.
    """

    def __init__(self, csv_path, device="cpu", log_params=False):
        self.df = pd.read_csv(csv_path)
        self.device = torch.device(device)

        X = torch.tensor(self.df[PARAM_COLS].values, dtype=torch.float32)
        y = torch.tensor(self.df[EFF_COLS].values, dtype=torch.float32)
        if log_params:
            X = X.clone()
            X[:, 3] = torch.log(X[:, 3])  # log period

        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0).clamp_min(1e-8)
        self.y_mean = y.mean(dim=0)
        self.y_std = y.std(dim=0).clamp_min(1e-8)

        self.X = (X - self.X_mean) / self.X_std
        self.y = (y - self.y_mean) / self.y_std
        self.y_raw_mean = self.y_mean
        self.y_raw_std = self.y_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def denormalize_y(self, y_norm):
        return y_norm * self.y_std.to(y_norm.device) + self.y_mean.to(y_norm.device)