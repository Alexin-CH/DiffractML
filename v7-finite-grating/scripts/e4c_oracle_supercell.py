"""E4c: at nh=8, can a *learned* low-order supercell profile beat the ideal cell?

The single-period oracle (E4a) cannot represent a localised missing ridge.  A
padded supercell can, but at nh=8 the *true* pattern is badly truncated
(Experiment 1: 0.0339, worse than the ideal cell's 0.0233).  The open question
is whether a profile **optimised at nh=8** — allowed to differ from the true
one — can beat both.

This is the existence test for the learned low-order supercell, before any
encoder is trained.

Example
-------
    python scripts/e4c_oracle_supercell.py --specs 0 5 13 22 --iters 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from m2_data import make_spec  # noqa: E402
from rcwa_supercell_diff import (  # noqa: E402
    profile_from_controls,
    solve_profile,
    true_profile,
)

MAX_HEIGHT_UM = 0.20
PADDING_UM = 1.0


def parse(argv=None):
    p = argparse.ArgumentParser(description="E4c oracle low-order supercell")
    p.add_argument("--data", default=os.path.join(
        _HERE, "..", "reports", "m2", "dataset_ideal_nh8.npz"))
    p.add_argument("--super-data", default=os.path.join(
        _HERE, "..", "reports", "m2", "dataset.npz"))
    p.add_argument("--specs", type=int, nargs="*", default=None)
    p.add_argument("--controls", type=int, default=24)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--n-fit-wl", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.10)
    p.add_argument("--nh", type=int, default=8)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=40)
    p.add_argument("--pol", choices=["s", "p"], default="s")
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=os.path.join(_HERE, "..", "reports", "m2", "e4c"))
    p.add_argument("--tag", default="oracle")
    return p.parse_args(argv)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _controls_init(spec, Lx, k):
    prof = true_profile(spec, Lx, 4096)
    x = (torch.arange(k, dtype=torch.float32) + 0.5) * (Lx / k)
    idx = (x / Lx * 4096).long() % 4096
    return prof[idx]


def _solve_all(h_fn, spec, Lx, wl, a, tin):
    dev = torch.device(a.device)
    rr, tt = [], []
    with torch.no_grad():
        for w in wl:
            h = h_fn()
            R, T = solve_profile(
                h, Lx, float(w), nh=a.nh, nx=a.nx, n_layers=a.n_layers,
                pol=a.pol, substrate_index=spec.substrate_index,
                device=a.device, tin_params=tin, tin_dataset=a.tin_dataset,
            )
            rr.append(float(R))
            tt.append(float(T))
    return np.asarray(rr), np.asarray(tt)


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(0)
    d = np.load(a.data, allow_pickle=True)
    ds = np.load(a.super_data, allow_pickle=True)
    wl = d["wl"].astype(np.float64)
    group = d["group"].astype(np.int64)
    y_abs = d["Y_abs"].astype(np.float64)
    ideal_rt = d["RCWA_abs"].astype(np.float64)
    super32 = ds["RCWA_abs"].astype(np.float64)

    from tin_dispersion import get_or_fit

    tin = get_or_fit(a.tin_dataset,
                     cache_path=os.path.join(_HERE, "..", "data",
                                             f"tin_fit_{a.tin_dataset}.json"))

    ids = sorted(np.unique(group).tolist())
    if a.specs:
        ids = [i for i in ids if i in set(a.specs)]

    rows = []
    t0 = time.time()
    for k, sid in enumerate(ids):
        m = group == sid
        wl_s = wl[m]
        Y = y_abs[m]
        spec = make_spec(sid)
        Lx = spec.n_ridges * spec.period + 2.0 * PADDING_UM
        dev = torch.device(a.device)

        ideal_A = 1.0 - ideal_rt[m, 0] - ideal_rt[m, 1]
        super32_A = 1.0 - super32[m, 0] - super32[m, 1]

        # true pattern truncated at nh (the baseline this must beat)
        h_true_ctrl = _controls_init(spec, Lx, a.controls).to(dev)
        def h_true():
            return profile_from_controls(
                h_true_ctrl.clamp(min=0.0), a.nx, Lx)
        r_nh8, t_nh8 = _solve_all(h_true, spec, Lx, wl_s, a, tin)
        nh8_A = 1.0 - r_nh8 - t_nh8

        fit_idx = np.linspace(0, len(wl_s) - 1, a.n_fit_wl).round().astype(int)
        R_t = torch.tensor(Y[fit_idx, 0], device=dev)
        T_t = torch.tensor(Y[fit_idx, 1], device=dev)

        # optimise control heights (sigmoid keeps 0 <= h <= MAX_HEIGHT_UM)
        u0 = torch.logit((h_true_ctrl / MAX_HEIGHT_UM).clamp(1e-3, 1 - 1e-3))
        inits = [u0.clone()]
        if a.restarts > 1:
            inits.append(u0.clone() * 0.5)
        best_params = None
        for u_init in inits[: a.restarts]:
            u = u_init.clone().requires_grad_(True)
            opt = torch.optim.Adam([u], lr=a.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
            best_here = None
            for _ in range(a.iters):
                h_ctrl = MAX_HEIGHT_UM * torch.sigmoid(u)
                h = profile_from_controls(h_ctrl, a.nx, Lx)
                loss = torch.zeros((), device=dev)
                for j, wj in enumerate(wl_s[fit_idx]):
                    R, T = solve_profile(
                        h, Lx, float(wj), nh=a.nh, nx=a.nx,
                        n_layers=a.n_layers, pol=a.pol,
                        substrate_index=spec.substrate_index,
                        device=a.device, tin_params=tin,
                        tin_dataset=a.tin_dataset,
                    )
                    loss = loss + (R - R_t[j]) ** 2 + (T - T_t[j]) ** 2
                loss = loss / len(fit_idx)
                if best_here is None or float(loss.detach()) < best_here[1]:
                    with torch.no_grad():
                        best_here = (h_ctrl.detach().clone(),
                                     float(loss.detach()))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if best_params is None or best_here[1] < best_params[1]:
                best_params = best_here

        # candidates: optimised profile and the true pattern; pick by the
        # full (R, T) response — matching A alone can be a degenerate win.
        cands = [best_params[0], h_true_ctrl.clamp(min=0.0)]
        best = None
        for h_ctrl in cands:
            def h_fn(h_ctrl=h_ctrl):
                return profile_from_controls(h_ctrl, a.nx, Lx)
            r, t = _solve_all(h_fn, spec, Lx, wl_s, a, tin)
            mse = float(np.mean((r - Y[:, 0]) ** 2 + (t - Y[:, 1]) ** 2))
            if best is None or mse < best[1]:
                best = (h_ctrl, mse, r, t)

        h_ctrl, full_mse, r_or, t_or = best
        or_A = 1.0 - r_or - t_or
        prof = profile_from_controls(h_ctrl, a.nx, Lx)
        row = {
            "spec": sid,
            "n_missing": int(np.asarray(spec.missing, dtype=bool).sum()),
            "period": float(spec.period),
            "Lx": float(Lx),
            "rmse_ideal": _rmse(ideal_A, Y[:, 2]),
            "rmse_super_nh8": _rmse(nh8_A, Y[:, 2]),
            "rmse_oracle_nh8": _rmse(or_A, Y[:, 2]),
            "rmse_super_nh32": _rmse(super32_A, Y[:, 2]),
            "R_ideal": _rmse(ideal_rt[m, 0], Y[:, 0]),
            "T_ideal": _rmse(ideal_rt[m, 1], Y[:, 1]),
            "R_super_nh8": _rmse(r_nh8, Y[:, 0]),
            "T_super_nh8": _rmse(t_nh8, Y[:, 1]),
            "R_oracle_nh8": _rmse(r_or, Y[:, 0]),
            "T_oracle_nh8": _rmse(t_or, Y[:, 1]),
            "R_super_nh32": _rmse(super32[m, 0], Y[:, 0]),
            "T_super_nh32": _rmse(super32[m, 1], Y[:, 1]),
            "profile_peak_um": float(prof.max()),
            "profile_mean_um": float(prof.mean()),
            "full_mse_RT": full_mse,
        }
        rows.append(row)
        print(f"[{k + 1}/{len(ids)}] spec {sid:03d} miss={row['n_missing']} "
              f"A ideal={row['rmse_ideal']:.4f} true@nh8={row['rmse_super_nh8']:.4f} "
              f"oracle@nh8={row['rmse_oracle_nh8']:.4f} "
              f"super@nh32={row['rmse_super_nh32']:.4f} | "
              f"R/T oracle={row['R_oracle_nh8']:.3f}/{row['T_oracle_nh8']:.3f} "
              f"super32={row['R_super_nh32']:.3f}/{row['T_super_nh32']:.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    def agg(key):
        v = [r[key] for r in rows]
        return float(np.mean(v)), float(np.std(v))

    os.makedirs(a.out_dir, exist_ok=True)
    out = {"config": vars(a), "per_spec": rows,
           "summary": {k: agg(k) for k in
                       ("rmse_ideal", "rmse_super_nh8",
                        "rmse_oracle_nh8", "rmse_super_nh32")}}
    with open(os.path.join(a.out_dir, f"{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    lines = ["# E4c — oracle low-order supercell (nh=8)", "",
             f"- geometries: {len(rows)}; profile: {a.controls} control "
             f"points, 0–{MAX_HEIGHT_UM * 1000:.0f} nm",
             f"- cell: aperture + {PADDING_UM} µm padding; nx={a.nx}, "
             f"{a.n_layers} layers",
             f"- solver: nh={a.nh}; optimiser: Adam, {a.iters} steps × "
             f"{a.restarts} restarts; selection on the full (R, T) response",
             "",
             "## Raw RMSE(A)", "",
             "| spec | miss | p (µm) | ideal | true @nh8 | oracle @nh8 | super @nh32 |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['spec']:03d} | {r['n_missing']} | {r['period']:.3f} | "
                     f"{r['rmse_ideal']:.4f} | {r['rmse_super_nh8']:.4f} | "
                     f"{r['rmse_oracle_nh8']:.4f} | {r['rmse_super_nh32']:.4f} |")
    s = out["summary"]
    lines += ["",
              "| mean | | | "
              f"{s['rmse_ideal'][0]:.4f} | {s['rmse_super_nh8'][0]:.4f} | "
              f"{s['rmse_oracle_nh8'][0]:.4f} | {s['rmse_super_nh32'][0]:.4f} |", "",
              "## Raw RMSE(R) / RMSE(T)", "",
              "| spec | ideal | true @nh8 | oracle @nh8 | super @nh32 |",
              "|---:|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['spec']:03d} | {r['R_ideal']:.3f}/{r['T_ideal']:.3f} | "
            f"{r['R_super_nh8']:.3f}/{r['T_super_nh8']:.3f} | "
            f"{r['R_oracle_nh8']:.3f}/{r['T_oracle_nh8']:.3f} | "
            f"{r['R_super_nh32']:.3f}/{r['T_super_nh32']:.3f} |")
    lines.append("")
    path = os.path.join(a.out_dir, f"{a.tag}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\nraw RMSE(A), mean over geometries")
    for key in ("rmse_ideal", "rmse_super_nh8", "rmse_oracle_nh8",
                "rmse_super_nh32"):
        print(f"  {key:18s} {s[key][0]:.4f}")
    print(f"wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
