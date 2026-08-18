"""Experiment matrix comparing inverse-design modes on the DiffractML v6 pipeline.

Modes (per target structure):
  1. surrogate         - surrogate-only design (no TORCWA)
  2. surrogate+amp     - surrogate init, then TORCWA refine amp only
  3. surrogate+ampper  - surrogate init, then TORCWA refine amp AND per
  4. solver-only       - no surrogate; TORCWA from random restarts (refine amp+per)

All runs: same wl-grid, ang, nh, discretization, iters, on one device.
Collects: recovered amp, per, final spectrum loss, wall time.
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


def spectrum_loss(wl_grid, ang, amp, per, targets, device):
    with torch.no_grad():
        preds = []
        for wl in wl_grid:
            args = SinTinArgs(wl=wl, ang=ang, nh=NH, discretization=DISC,
                              sin_amplitude=amp, sin_period=per,
                              uni_layer_h=30.0, device=device)
            eff = get_power_efficiencies(setup(args))
            preds.append(torch.stack([eff[c] for c in EFF_COLS]))
        pred = torch.stack(preds)
    return torch.nn.functional.mse_loss(pred, targets).item()


def main():
    device = pick_device(NH, DISC, "auto")
    print(f"device={device}", flush=True)
    sd = SurrogateDesign(SURROGATE)
    rows = []
    for amp_t, per_t in TARGETS:
        targets = build_targets(WL_GRID, ANG, amp_t, per_t, device)
        print(f"=== target amp={amp_t}, per={per_t} ===", flush=True)

        # 1. surrogate only
        t0 = time.time()
        amp_s, per_s, loss_s = sd.design(WL_GRID, ANG, targets)
        loss_s_rcwa = spectrum_loss(WL_GRID, ANG, amp_s, per_s, targets, device)
        rows.append(("surrogate", amp_t, per_t, amp_s, per_s,
                     loss_s_rcwa, time.time() - t0))
        print(f"  surrogate: amp={amp_s:.1f} per={per_s:.1f} loss(rcwa)={loss_s_rcwa:.6f}",
              flush=True)

        # 2. surrogate + refine amp
        t0 = time.time()
        amp2, per2, _ = design_with_torcwa_spectrum(
            WL_GRID, ANG, targets, amp_s, per_s, nh=NH, discretization=DISC,
            iters=ITERS, device=device, refine_per=False)
        loss2 = spectrum_loss(WL_GRID, ANG, amp2, per2, targets, device)
        rows.append(("surrogate+amp", amp_t, per_t, amp2, per2,
                     loss2, time.time() - t0))
        print(f"  surrogate+amp: amp={amp2:.1f} per={per2:.1f} loss={loss2:.6f}",
              flush=True)

        # 3. surrogate + refine amp+per
        t0 = time.time()
        amp3, per3, _ = design_with_torcwa_spectrum(
            WL_GRID, ANG, targets, amp_s, per_s, nh=NH, discretization=DISC,
            iters=ITERS, device=device, refine_per=True, per_lr=REFINE_PER_LR)
        loss3 = spectrum_loss(WL_GRID, ANG, amp3, per3, targets, device)
        rows.append(("surrogate+ampper", amp_t, per_t, amp3, per3,
                     loss3, time.time() - t0))
        print(f"  surrogate+ampper: amp={amp3:.1f} per={per3:.1f} loss={loss3:.6f}",
              flush=True)

        # 4. solver-only random restarts (refine amp+per)
        t0 = time.time()
        best = None
        for _ in range(RESTARTS):
            amp0 = torch.rand(1).mul(AMP_RANGE[1] - AMP_RANGE[0]).add(AMP_RANGE[0]).item()
            per0 = torch.rand(1).mul(PER_RANGE[1] - PER_RANGE[0]).add(PER_RANGE[0]).item()
            amp_r, per_r, loss_r = design_with_torcwa_spectrum(
                WL_GRID, ANG, targets, amp0, per0, nh=NH, discretization=DISC,
                iters=ITERS, device=device, refine_per=True, per_lr=REFINE_PER_LR)
            if best is None or loss_r < best[0]:
                best = (loss_r, amp_r, per_r)
        loss_r, amp_r, per_r = best
        loss_r_rcwa = spectrum_loss(WL_GRID, ANG, amp_r, per_r, targets, device)
        rows.append(("solver-only", amp_t, per_t, amp_r, per_r,
                     loss_r_rcwa, time.time() - t0))
        print(f"  solver-only: amp={amp_r:.1f} per={per_r:.1f} loss={loss_r_rcwa:.6f}",
              flush=True)

    print("\nMODE,target_amp,target_per,amp,per,loss,time_s")
    for r in rows:
        print(",".join(map(str, r)))
    with open("/tmp/opencode/exp_report/results.csv", "w") as f:
        f.write("mode,target_amp,target_per,amp,per,loss,time_s\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


if __name__ == "__main__":
    main()
