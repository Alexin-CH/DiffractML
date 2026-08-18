"""Experiment matrix comparing Adam vs LBFGS RCWA refinement.

All runs use the surrogate init (surrogate -> refine amp+per) and solver-only
modes, but with optimizer in {adam, lbfgs}.  Same config as the first report.
"""

import sys
import time

sys.path.insert(0, "/home/alexin/Documents/Academic/Research/DiffractML/v6-inverse-design/src")

import torch

from dataset import EFF_COLS
from design import (
    AMP_RANGE,
    PER_RANGE,
    SurrogateDesign,
    design_with_torcwa_spectrum,
    pick_device,
)
from simulation import SinTinArgs, get_power_efficiencies, setup

SURROGATE = "/home/alexin/Documents/Academic/Research/DiffractML/v6-inverse-design/model_pt/surrogate.pt"

WL_GRID = [800.0, 1100.0, 1400.0, 1600.0]
ANG = 30.0
NH = 8
DISC = 48
ITERS = 25
REFINE_PER_LR = 1.0
RESTARTS = 3
LBFGS_LR = 1.0

TARGETS = [(45.0, 1500.0), (35.0, 1000.0), (60.0, 2500.0)]


def build_targets(wl_grid, ang, amp_t, per_t, device):
    tgt = []
    for wl in wl_grid:
        args = SinTinArgs(wl=wl, ang=ang, nh=NH, discretization=DISC,
                          sin_amplitude=amp_t, sin_period=per_t,
                          uni_layer_h=30.0, device=device)
        eff = get_power_efficiencies(setup(args))
        tgt.append([eff[c].item() for c in EFF_COLS])
    return torch.tensor(tgt, device=device)


def main():
    device = pick_device(NH, DISC, "auto")
    print(f"device={device}", flush=True)
    sd = SurrogateDesign(SURROGATE)
    rows = []
    for amp_t, per_t in TARGETS:
        targets = build_targets(WL_GRID, ANG, amp_t, per_t, device)
        print(f"=== target amp={amp_t}, per={per_t} ===", flush=True)

        for opt_name, opt_kwargs in [("adam", {}), ("lbfgs", {"lr": LBFGS_LR})]:
            # surrogate + refine amp+per
            amp_s, per_s, _ = sd.design(WL_GRID, ANG, targets)
            t0 = time.time()
            amp_r, per_r, loss_r = design_with_torcwa_spectrum(
                WL_GRID, ANG, targets, amp_s, per_s, nh=NH, discretization=DISC,
                iters=ITERS, device=device, refine_per=True,
                per_lr=REFINE_PER_LR, optimizer=opt_name, **opt_kwargs)
            dt = time.time() - t0
            rows.append((f"surrogate+{opt_name}", amp_t, per_t,
                         amp_r, per_r, loss_r, dt))
            print(f"  surrogate+{opt_name}: amp={amp_r:.1f} per={per_r:.1f} "
                  f"loss={loss_r:.6f} t={dt:.1f}s", flush=True)

            # solver-only from the SAME random starts for fairness
            t0 = time.time()
            best = None
            for _ in range(RESTARTS):
                amp0 = torch.rand(1).mul(AMP_RANGE[1] - AMP_RANGE[0]).add(AMP_RANGE[0]).item()
                per0 = torch.rand(1).mul(PER_RANGE[1] - PER_RANGE[0]).add(PER_RANGE[0]).item()
                amp_r2, per_r2, loss_r2 = design_with_torcwa_spectrum(
                    WL_GRID, ANG, targets, amp0, per0, nh=NH, discretization=DISC,
                    iters=ITERS, device=device, refine_per=True,
                    per_lr=REFINE_PER_LR, optimizer=opt_name, **opt_kwargs)
                if best is None or loss_r2 < best[0]:
                    best = (loss_r2, amp_r2, per_r2)
            loss_r2, amp_r2, per_r2 = best
            dt = time.time() - t0
            rows.append((f"solver-only-{opt_name}", amp_t, per_t,
                         amp_r2, per_r2, loss_r2, dt))
            print(f"  solver-only-{opt_name}: amp={amp_r2:.1f} per={per_r2:.1f} "
                  f"loss={loss_r2:.6f} t={dt:.1f}s", flush=True)

    print("\nMODE,target_amp,target_per,amp,per,loss,time_s")
    for r in rows:
        print(",".join(map(str, r)))
    with open("/tmp/opencode/exp_report/results_lbfgs.csv", "w") as f:
        f.write("mode,target_amp,target_per,amp,per,loss,time_s\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


if __name__ == "__main__":
    main()