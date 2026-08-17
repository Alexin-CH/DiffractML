"""Inverse design: given a target optical response, recover (amp, per).

Two strategies:
  1. Fast surrogate-based design: gradient descent through the trained NN
     surrogate (milliseconds).
  2. TORCWA-gradient refinement: gradient descent through the differentiable
     RCWA solver itself (seconds), to refine/validate the surrogate result.
"""

import argparse
import os

import torch

from dataset import EFF_COLS, PARAM_COLS, SinTinDataset
from models import SurrogateMLP
from simulation import SinTinArgs, get_power_efficiencies, setup

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Physical bounds for the design variables (nm)
AMP_RANGE = (20.0, 100.0)
PER_RANGE = (200.0, 5000.0)


class SurrogateDesign:
    """Optimize (amp, per) through the surrogate to match a target response."""

    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model = SurrogateMLP(
            in_dim=len(PARAM_COLS),
            hidden=ckpt["args"]["hidden"],
            out_dim=len(EFF_COLS),
            depth=ckpt["args"]["depth"],
        )
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.X_mean = ckpt["X_mean"]
        self.X_std = ckpt["X_std"]
        self.y_mean = ckpt["y_mean"]
        self.y_std = ckpt["y_std"]

    def _normalize(self, wl, ang, amp, per):
        x = torch.stack([
            torch.as_tensor(v, dtype=torch.float32).reshape(-1)
            for v in (wl, ang, amp, per)
        ]).reshape(-1)
        return (x - self.X_mean) / self.X_std

    def _predict(self, wl, ang, amp, per):
        x = self._normalize(wl, ang, amp, per).unsqueeze(0)
        with torch.no_grad():
            y_n = self.model(x)
        return (y_n * self.y_std + self.y_mean).squeeze(0)

    def design(self, wl_grid, ang, targets, n_restarts=20, iters=300, lr=0.5):
        """Minimize ||pred(wl_grid, ang, amp, per) - targets||^2 over (amp, per).

        ``wl_grid`` and ``targets`` are arrays over a wavelength range; using
        a spectrum (rather than a single wavelength) makes the inverse
        problem well-posed.  ``targets`` shape (n_wl, 4) for [R_xx, T_xx,
        R_yy, T_yy].
        """
        wl_grid = torch.tensor(wl_grid, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        best = None
        for r in range(n_restarts):
            amp = torch.rand(1) * (AMP_RANGE[1] - AMP_RANGE[0]) + AMP_RANGE[0]
            per = torch.rand(1) * (PER_RANGE[1] - PER_RANGE[0]) + PER_RANGE[0]
            amp.requires_grad_()
            per.requires_grad_()
            opt = torch.optim.Adam([amp, per], lr=lr)
            for _ in range(iters):
                opt.zero_grad()
                pred = self._predict_spectrum(wl_grid, ang, amp, per)
                loss = torch.nn.functional.mse_loss(pred, targets)
                loss.backward()
                opt.step()
                amp.data.clamp_(*AMP_RANGE)
                per.data.clamp_(*PER_RANGE)
            loss = self._loss_spectrum(wl_grid, ang, amp.item(), per.item(), targets)
            if best is None or loss < best[0]:
                best = (loss.item(), amp.item(), per.item())
        return best[1], best[2], best[0]

    def _predict_spectrum(self, wl_grid, ang, amp, per):
        """Predict the full spectrum for one (amp, per) — differentiable."""
        x = torch.stack([self._normalize(wl, ang, amp, per) for wl in wl_grid])
        y_n = self.model(x)
        return y_n * self.y_std + self.y_mean

    def _loss_spectrum(self, wl_grid, ang, amp, per, targets):
        with torch.no_grad():
            pred = self._predict_spectrum(wl_grid, ang, amp, per)
        return torch.nn.functional.mse_loss(pred, targets)


def design_with_torcwa_spectrum(wl_grid, ang, targets, amp_init, per_init,
                                nh=8, discretization=48, iters=60, lr=5.0,
                                device="cpu", edge_sharpness=50.0):
    """Refine ``amp`` against a target spectrum through RCWA gradients.

    ``targets`` shape (n_wl, 4).  ``per`` is fixed (its gradient is weak in
    TORCWA).  Returns refined (amp, per).
    """
    amp = torch.tensor(float(amp_init), requires_grad=True)
    targets = torch.tensor(targets).to(torch.float32)

    opt = torch.optim.Adam([amp], lr=lr)
    for i in range(iters):
        opt.zero_grad()
        preds = []
        for wl in wl_grid:
            args = SinTinArgs(
                wl=wl, ang=ang, nh=nh, discretization=discretization,
                sin_amplitude=amp, sin_period=per_init, uni_layer_h=30.0,
                edge_sharpness=edge_sharpness, device=device,
            )
            sim = setup(args)
            eff = get_power_efficiencies(sim)
            preds.append(torch.stack([eff[c] for c in EFF_COLS]))
        pred = torch.stack(preds)
        loss = torch.nn.functional.mse_loss(pred, targets)
        if not torch.isfinite(loss):
            break
        loss.backward()
        if amp.grad is None or not torch.isfinite(amp.grad):
            break
        opt.step()
        amp.data.clamp_(30.0, 80.0)
    return amp.item(), per_init


def design_with_torcwa(wl, ang, target, amp_init, per_init, nh=8,
                       discretization=48, iters=60, lr=5.0, device="cpu",
                       edge_sharpness=50.0):
    """Refine ``amp`` by backpropagating through the RCWA solver.

    The period gradient is effectively zero in TORCWA (the lattice and
    k-vectors are built from detached scalars), so only ``amp`` is refined
    here; ``per`` is supplied by the surrogate.  Returns the refined
    (amp, per) and the achieved response.
    """
    amp = torch.tensor(float(amp_init), requires_grad=True)
    target = torch.tensor(target, dtype=torch.float32)

    opt = torch.optim.Adam([amp], lr=lr)
    for i in range(iters):
        opt.zero_grad()
        args = SinTinArgs(
            wl=wl, ang=ang, nh=nh, discretization=discretization,
            sin_amplitude=amp, sin_period=per_init, uni_layer_h=30.0,
            edge_sharpness=edge_sharpness, device=device,
        )
        sim = setup(args)
        eff = get_power_efficiencies(sim)
        pred = torch.stack([eff[c] for c in EFF_COLS])
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        amp.data.clamp_(*AMP_RANGE)

    with torch.no_grad():
        args = SinTinArgs(
            wl=wl, ang=ang, nh=nh, discretization=discretization,
            sin_amplitude=amp, sin_period=per_init, uni_layer_h=30.0,
            edge_sharpness=edge_sharpness, device=device,
        )
        sim = setup(args)
        eff = get_power_efficiencies(sim)
        pred = torch.stack([eff[c] for c in EFF_COLS])
    return amp.item(), per_init, pred.detach()


def parse():
    p = argparse.ArgumentParser(description="Inverse design for sine TiN grating")
    p.add_argument("--surrogate", type=str, default=os.path.join(CURRENT_DIR, "..", "model_pt", "surrogate.pt"))
    p.add_argument("--wl-grid", type=float, nargs="+", default=None,
                   help="wavelength grid (nm). Default: 800..1600")
    p.add_argument("--ang", type=float, default=0.0, help="incidence angle (deg)")
    p.add_argument("--target-amp", type=float, default=45.0,
                   help="ground-truth amplitude for self-check target generation")
    p.add_argument("--target-per", type=float, default=1500.0,
                   help="ground-truth period for self-check target generation")
    p.add_argument("--target-col", type=str, default="R_xx",
                   help="efficiency column to match (R_xx, T_xx, R_yy, T_yy)")
    p.add_argument("--nh", type=int, default=8)
    p.add_argument("--discretization", type=int, default=48)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    a = parse()
    wl_grid = a.wl_grid or list(range(800, 1600, 100))
    col_idx = EFF_COLS.index(a.target_col)

    # Ground-truth target spectrum computed with TORCWA (the "spec" we want).
    from simulation import SinTinArgs as _SA, setup as _setup, get_power_efficiencies as _gpe
    targets = []
    for wl in wl_grid:
        args = _SA(wl=wl, ang=a.ang, nh=a.nh, discretization=a.discretization,
                   sin_amplitude=a.target_amp, sin_period=a.target_per,
                   uni_layer_h=30.0, device=a.device)
        sim = _setup(args)
        eff = _gpe(sim)
        targets.append([eff[c].item() for c in EFF_COLS])
    targets = torch.tensor(targets)
    print(f"Ground truth structure: amp={a.target_amp} nm, per={a.target_per} nm")
    print(f"Target {a.target_col} spectrum: "
          f"{[round(float(t),4) for t in targets[:, col_idx].tolist()]}")

    sd = SurrogateDesign(a.surrogate)
    amp, per, loss = sd.design(wl_grid, a.ang, targets)
    print(f"[surrogate] designed amp={amp:.1f} nm, per={per:.1f} nm  (loss={loss:.5f})")
    pred = sd._predict_spectrum(wl_grid, a.ang, amp, per).detach()
    print(f"[surrogate] predicted {a.target_col} spectrum: "
          f"{[round(float(x),4) for x in pred[:, col_idx].tolist()]}")

    print(f"[torcwa]    refining amp through RCWA gradients ...")
    amp2, per2 = design_with_torcwa_spectrum(
        wl_grid, a.ang, targets, amp, per, nh=a.nh,
        discretization=a.discretization, iters=a.iters, device=a.device,
    )
    print(f"[torcwa]    refined amp={amp2:.1f} nm (truth {a.target_amp:.1f}), per={per2:.1f} (truth {a.target_per:.1f})")