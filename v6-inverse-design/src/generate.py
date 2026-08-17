"""Dataset generation for v6 inverse design.

Generates samples over the sine TiN structure parameter space using TORCWA,
correctly extracting per-polarization S-parameters and total power
efficiencies (fixes the archive/v5 bug where all 32 S-parameter columns were
identical).

Output: a CSV with columns
    wl, ang, amp, per,
    + forward/backward transmission/reflection for each polarization
      (complex S-parameter, real part from ``.real``)
    + R_xx, T_xx, R_yy, T_yy (total power efficiencies)
"""

import argparse
import os
import time

import pandas as pd
import torch
from tqdm import tqdm

from simulation import (
    S_PARAM_KEYS,
    SinTinArgs,
    get_power_efficiencies,
    get_s_parameters,
    setup,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse():
    p = argparse.ArgumentParser(description="Generate v6 inverse-design dataset")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--nh", type=int, default=10)
    p.add_argument("--discretization", type=int, default=64)
    p.add_argument("--uni-layer-h", type=float, default=30.0)
    p.add_argument("--wl-range", type=float, nargs=2, default=[400.0, 2000.0])
    p.add_argument("--ang-range", type=float, nargs=2, default=[0.0, 70.0])
    p.add_argument("--amp-range", type=float, nargs=2, default=[20.0, 100.0])
    p.add_argument("--per-range", type=float, nargs=2, default=[200.0, 5000.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def sample_parameters(args):
    """Draw the parameter grid (deterministic given the seed)."""
    g = torch.Generator().manual_seed(args.seed)
    wl = torch.rand(args.n_samples, generator=g) * (args.wl_range[1] - args.wl_range[0]) + args.wl_range[0]
    ang = torch.rand(args.n_samples, generator=g) * (args.ang_range[1] - args.ang_range[0]) + args.ang_range[0]
    amp = torch.rand(args.n_samples, generator=g) * (args.amp_range[1] - args.amp_range[0]) + args.amp_range[0]
    # log-spaced period
    log_min = torch.log(torch.tensor(args.per_range[0]))
    log_max = torch.log(torch.tensor(args.per_range[1]))
    per = torch.exp(torch.rand(args.n_samples, generator=g) * (log_max - log_min) + log_min)
    return wl, ang, amp, per


def generate(args):
    t0 = time.time()
    wl, ang, amp, per = sample_parameters(args)

    rows = []
    for i in tqdm(range(args.n_samples), desc="RCWA samples"):
        sim_args = SinTinArgs(
            wl=wl[i].item(),
            ang=ang[i].item(),
            nh=args.nh,
            discretization=args.discretization,
            sin_amplitude=amp[i].item(),
            sin_period=per[i].item(),
            uni_layer_h=args.uni_layer_h,
            device=args.device,
        )
        sim = setup(sim_args)

        row = {
            "wl": wl[i].item(),
            "ang": ang[i].item(),
            "amp": amp[i].item(),
            "per": per[i].item(),
        }
        for k, v in get_s_parameters(sim).items():
            row[k + ".real"] = v.item().real
            row[k + ".imag"] = v.item().imag
        for k, v in get_power_efficiencies(sim).items():
            row[k] = v.item()

        rows.append(row)

    df = pd.DataFrame(rows)
    if args.output is None:
        args.output = os.path.join(CURRENT_DIR, "..", "data", "dataset_v6.csv")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} samples to {args.output} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    generate(parse())