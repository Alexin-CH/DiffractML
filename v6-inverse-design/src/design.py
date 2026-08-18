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

# RCWA solve crossover: below this problem size CPU is faster (kernel/transfer
# overhead dominates), above it CUDA wins.  Estimated from a CPU-vs-CUDA
# sweep: GPU is ~3x faster at nh>=8/disc>=48, ~2x slower below.
CUDA_MIN_NH = 8
CUDA_MIN_DISC = 48


def pick_device(nh, discretization, requested="auto"):
    """Choose the fastest device for an RCWA problem size.

    ``requested``: 'cpu', 'cuda', or 'auto' (hybrid: CUDA only where it is
    faster).  Returns 'cuda' iff CUDA is available and the problem is large
    enough for GPU speedup to overcome launch overhead.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    use_cuda = nh >= CUDA_MIN_NH and discretization >= CUDA_MIN_DISC
    return "cuda" if use_cuda else "cpu"


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
        wl_grid = torch.as_tensor(wl_grid, dtype=torch.float32, device="cpu")
        targets = torch.as_tensor(targets, dtype=torch.float32, device="cpu")
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
                                device="cpu", edge_sharpness=50.0,
                                refine_per=False, per_lr=1.0):
    """Refine ``amp`` (and optionally ``per``) against a target spectrum
    through RCWA gradients.

    ``targets`` shape (n_wl, 4).  ``per`` is fixed by default: with the
    detached-lattice setup its RCWA gradient is physically near-zero.  With
    ``refine_per=True`` the lattice/grid are built from the period tensor
    (``tensor_lattice``) so ``per`` also receives a physically meaningful
    gradient.  Returns refined (amp, per).
    """
    amp = torch.tensor(float(amp_init), device=device, requires_grad=True)
    per = torch.tensor(float(per_init), device=device, requires_grad=True)
    targets = torch.tensor(targets, device=device).to(torch.float32)

    if refine_per:
        opt = torch.optim.Adam(
            [{"params": [amp], "lr": lr}, {"params": [per], "lr": per_lr}]
        )
    else:
        opt = torch.optim.Adam([amp], lr=lr)

    for i in range(iters):
        opt.zero_grad()
        preds = []
        for wl in wl_grid:
            args = SinTinArgs(
                wl=wl, ang=ang, nh=nh, discretization=discretization,
                sin_amplitude=amp, sin_period=per, uni_layer_h=30.0,
                edge_sharpness=edge_sharpness, device=device,
            )
            sim = setup(args, tensor_lattice=refine_per)
            eff = get_power_efficiencies(sim)
            preds.append(torch.stack([eff[c] for c in EFF_COLS]))
        pred = torch.stack(preds)
        loss = torch.nn.functional.mse_loss(pred, targets)
        if not torch.isfinite(loss):
            break
        loss.backward()
        if amp.grad is None or not torch.isfinite(amp.grad):
            break
        if refine_per and (per.grad is None or not torch.isfinite(per.grad)):
            break
        opt.step()
        amp.data.clamp_(30.0, 80.0)
        if refine_per:
            per.data.clamp_(*PER_RANGE)
    return amp.item(), per.item()


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
                   help="wavelength grid (nm). Default (cuda): 800..1600 step 100; "
                        "(cpu): [800, 1100, 1400, 1600]")
    p.add_argument("--ang", type=float, default=0.0, help="incidence angle (deg)")
    p.add_argument("--target-amp", type=float, default=45.0,
                   help="ground-truth amplitude for self-check target generation")
    p.add_argument("--target-per", type=float, default=1500.0,
                   help="ground-truth period for self-check target generation")
    p.add_argument("--target-col", type=str, default="R_xx",
                   help="efficiency column to match (R_xx, T_xx, R_yy, T_yy)")
    p.add_argument("--nh", type=int, default=None,
                   help="Fourier order. Default (cuda): 8; (cpu): 5")
    p.add_argument("--discretization", type=int, default=None,
                   help="spatial grid per axis. Default (cuda): 48; (cpu): 32")
    p.add_argument("--iters", type=int, default=None,
                   help="RCWA refinement iterations. Default (cuda): 60; (cpu): 25")
    p.add_argument("--device", type=str, default="auto",
                   help="device for RCWA solves: 'cpu', 'cuda', or 'auto' "
                        "(hybrid — CUDA only where it beats CPU)")
    p.add_argument("--refine-per", action="store_true",
                   help="also refine the period through RCWA gradients "
                        "(requires tensor-lattice setup)")
    p.add_argument("--per-lr", type=float, default=1.0,
                   help="learning rate for the period during RCWA refinement")
    return p.parse_args()


if __name__ == "__main__":
    a = parse()

    # Resolve problem size first: on 'auto', default to the full RCWA setup
    # (nh=8/disc=48) only when CUDA is available to run it fast; otherwise
    # fall back to the lighter CPU setup.  Explicit --nh/--disc always win.
    use_gpu = pick_device(
        a.nh if a.nh is not None else CUDA_MIN_NH,
        a.discretization if a.discretization is not None else CUDA_MIN_DISC,
        a.device,
    ) == "cuda"
    heavy = (a.device == "cuda") or (a.device == "auto" and use_gpu)

    wl_grid = a.wl_grid or (
        list(range(800, 1600, 100)) if heavy else [800.0, 1100.0, 1400.0, 1600.0]
    )
    nh = a.nh if a.nh is not None else (8 if heavy else 5)
    discretization = a.discretization if a.discretization is not None else (
        48 if heavy else 32
    )
    iters = a.iters if a.iters is not None else (60 if heavy else 25)
    rwa_device = pick_device(nh, discretization, a.device)
    col_idx = EFF_COLS.index(a.target_col)

    # Ground-truth target spectrum computed with TORCWA (the "spec" we want).
    from simulation import SinTinArgs as _SA, setup as _setup, get_power_efficiencies as _gpe
    targets = []
    for wl in wl_grid:
        args = _SA(wl=wl, ang=a.ang, nh=nh, discretization=discretization,
                   sin_amplitude=a.target_amp, sin_period=a.target_per,
                   uni_layer_h=30.0, device=rwa_device)
        sim = _setup(args)
        eff = _gpe(sim)
        targets.append([eff[c].item() for c in EFF_COLS])
    targets = torch.tensor(targets)
    print(f"Ground truth structure: amp={a.target_amp} nm, per={a.target_per} nm")
    print(f"RCWA device: {rwa_device} (requested: {a.device})")
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
        wl_grid, a.ang, targets, amp, per, nh=nh,
        discretization=discretization, iters=iters, device=rwa_device,
        refine_per=a.refine_per, per_lr=a.per_lr,
    )
    print(f"[torcwa]    refined amp={amp2:.1f} nm (truth {a.target_amp:.1f}), per={per2:.1f} (truth {a.target_per:.1f})")