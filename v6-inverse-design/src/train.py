"""Training script for the surrogate forward model."""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import EFF_COLS, PARAM_COLS, SinTinDataset
from models import SurrogateMLP

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse():
    p = argparse.ArgumentParser(description="Train v6 surrogate")
    p.add_argument("--data", type=str, default=os.path.join(CURRENT_DIR, "..", "data", "dataset_v6.csv"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save", type=str, default=os.path.join(CURRENT_DIR, "..", "model_pt", "surrogate.pt"))
    return p.parse_args()


def main(a):
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    ds = SinTinDataset(a.data, device=device)
    print(f"Loaded {len(ds)} samples. Input: {PARAM_COLS}, Output: {EFF_COLS}")

    val_size = int(len(ds) * a.val_split)
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size)

    model = SurrogateMLP(in_dim=len(PARAM_COLS), hidden=a.hidden, out_dim=len(EFF_COLS), depth=a.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)

        if epoch % 20 == 0 or epoch == a.epochs - 1:
            model.eval()
            with torch.no_grad():
                vt, vn = 0.0, 0
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    vt += loss_fn(model(x), y).item() * len(x)
                    vn += len(x)
            print(f"epoch {epoch:4d}  train={tot/n:.6f}  val={vt/vn:.6f}")

    os.makedirs(os.path.dirname(a.save), exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "X_mean": ds.X_mean, "X_std": ds.X_std,
                "y_mean": ds.y_mean, "y_std": ds.y_std}, a.save)
    print(f"Saved to {a.save}")


if __name__ == "__main__":
    main(parse())