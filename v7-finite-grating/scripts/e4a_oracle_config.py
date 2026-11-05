"""E4a: does an equivalent periodic config exist?  Oracle fit, no network.

For each geometry we directly optimise a four-parameter effective unit cell
``(period, amplitude, base, fill)`` against the MEEP reference with the
differentiable RCWA.  This is the best a config-predicting encoder could do:

* oracle ~ fixed ideal cell  -> no useful periodic equivalent exists;
* oracle -> padded supercell -> a config exists and an encoder is worth
  training.

The ``fill`` knob (duty cycle) is the one Experiment 2/3 were missing: a
missing ridge is a coverage defect, which an amplitude/period change cannot
express.

Example
-------
    python scripts/e4a_oracle_config.py --device cuda
    python scripts/e4a_oracle_config.py --specs 0 1 2 --iters 4  # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from m2_data import make_spec  # noqa: E402
from rcwa_diff import solve_unit_cell  # noqa: E402

P_RANGE = (0.60, 1.60)
A_RANGE = (0.10, 2.50)
B_RANGE = (0.20, 2.00)
F_RANGE = (0.20, 1.00)


def parse(argv=None):
    p = argparse.ArgumentParser(description="E4a oracle effective config")
    p.add_argument("--ideal-data", default=os.path.join(
        _HERE, "..", "reports", "m2", "dataset_ideal_nh8.npz"))
    p.add_argument("--super-data", default=os.path.join(
        _HERE, "..", "reports", "m2", "dataset.npz"))
    p.add_argument("--specs", type=int, nargs="*", default=None,
                   help="subset of spec ids (default: all)")
    p.add_argument("--n-fit-wl", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--nh", type=int, default=8)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--n-layers", type=int, default=40)
    p.add_argument("--pol", choices=["s", "p"], default="s")
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=os.path.join(_HERE, "..", "reports", "m2", "e4a"))
    p.add_argument("--eps-knob", action="store_true",
                   help="add an effective TiN permittivity scale (E4b)")
    p.add_argument("--tag", default="oracle")
    return p.parse_args(argv)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _constrained(raw, init, lo, hi):
    """Multiplicative box constraint ``init * exp(clip(raw, log lo, log hi))``.

    ``raw`` is the unconstrained optimisation variable; ``init`` may be a
    tensor (period/amplitude/base) so gradients still flow.
    """
    return init * torch.exp(torch.clamp(raw, math.log(lo), math.log(hi)))


def fit_oracle(spec, wl_all, y_all, tin_params, a):
    """Optimise ``(log p, log amp, log base, logit fill)`` against MEEP."""
    dev = torch.device(a.device)
    p0 = torch.tensor(float(spec.period), device=dev)
    a0 = torch.tensor(float(spec.amplitude), device=dev)
    b0 = torch.tensor(float(spec.base_thickness), device=dev)

    fit_idx = np.linspace(0, len(wl_all) - 1, a.n_fit_wl).round().astype(int)
    wl_fit = [float(wl_all[j]) for j in fit_idx]
    R_t = torch.tensor(y_all[fit_idx, 0], device=dev)
    T_t = torch.tensor(y_all[fit_idx, 1], device=dev)

    inits = [(0.0, 0.0, 0.0, 2.0, 0.0)]
    if a.restarts > 1:
        inits.append((math.log(1.15), math.log(1.30), 0.0, 0.0, 0.0))
    if a.restarts > 2:
        inits.append((math.log(0.85), math.log(0.60), 0.0, 3.0, 0.0))

    candidates = []
    for lp0, la0, lb0, uf0, ue0 in inits[: a.restarts]:
        raw = [torch.tensor(v, device=dev, requires_grad=True)
               for v in (lp0, la0, lb0, uf0, ue0)]
        opt = torch.optim.Adam(raw, lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        best_here = None
        for _ in range(a.iters):
            period = _constrained(raw[0], p0, *P_RANGE)
            amp = _constrained(raw[1], a0, *A_RANGE)
            base = _constrained(raw[2], b0, *B_RANGE)
            fill = F_RANGE[0] + (F_RANGE[1] - F_RANGE[0]) * torch.sigmoid(raw[3])
            eps_scale = 1.0 + 0.6 * torch.tanh(raw[4]) if a.eps_knob else None
            loss = torch.zeros((), device=dev)
            for k, w in enumerate(wl_fit):
                R, T = solve_unit_cell(
                    period, amp, base, w, nh=a.nh, nx=a.nx,
                    n_layers=a.n_layers, edge_sharpness=100.0, fill=fill,
                    eps_scale=eps_scale, pol=a.pol,
                    substrate_index=spec.substrate_index,
                    device=a.device, tin_params=tin_params,
                    tin_dataset=a.tin_dataset,
                )
                loss = loss + (R - R_t[k]) ** 2 + (T - T_t[k]) ** 2
            loss = loss / len(wl_fit)
            # Adam can wander off a good initial point; keep the best seen.
            if best_here is None or float(loss.detach()) < best_here[1]:
                with torch.no_grad():
                    best_here = ((
                        float(period.detach()), float(amp.detach()),
                        float(base.detach()), float(fill.detach()),
                        float(eps_scale.detach()) if eps_scale is not None else 1.0,
                    ), float(loss.detach()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        candidates.append(best_here)

    # The nominal ideal cell is always a candidate, so the oracle can never
    # be worse than the baseline.  Selection is on the **absorption** spectrum
    # (the reported metric), not on (R, T): fitting R and T can trade off A.
    candidates.append(((float(spec.period), float(spec.amplitude),
                        float(spec.base_thickness), 1.0, 1.0), 0.0))
    best = None
    for params, fit_mse in candidates:
        r, t = evaluate(params, spec, wl_all, a, tin_params)
        A = 1.0 - r - t
        full_mse_A = float(np.mean((A - y_all[:, 2]) ** 2))
        if best is None or full_mse_A < best[1]:
            best = (params, full_mse_A, fit_mse, r, t)
    return best, fit_idx


def evaluate(params, spec, wl_all, a, tin_params):
    """Raw RCWA (R, T) at every wavelength for fixed effective parameters."""
    dev = torch.device(a.device)
    period, amp, base, fill, eps_scale = params
    rr, tt = [], []
    with torch.no_grad():
        for w in wl_all:
            R, T = solve_unit_cell(
                torch.tensor(period, device=dev),
                torch.tensor(amp, device=dev),
                torch.tensor(base, device=dev),
                float(w), nh=a.nh, nx=a.nx, n_layers=a.n_layers,
                edge_sharpness=100.0, fill=fill, eps_scale=eps_scale,
                pol=a.pol,
                substrate_index=spec.substrate_index, device=a.device,
                tin_params=tin_params, tin_dataset=a.tin_dataset,
            )
            rr.append(float(R))
            tt.append(float(T))
    return np.asarray(rr), np.asarray(tt)


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = sorted(rows, key=lambda r: r["rmse_super"])
    x = np.arange(len(rs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.5), dpi=160)

    ax1.plot(x, [r["rmse_ideal"] for r in rs], "o-", ms=3.5, lw=1.2,
             color="#2d3142", label="ideal cell")
    ax1.plot(x, [r["rmse_oracle"] for r in rs], "s-", ms=3.5, lw=1.2,
             color="#bf4520", label="oracle config")
    ax1.plot(x, [r["rmse_super"] for r in rs], "^-", ms=3.5, lw=1.2,
             color="#2e5aa8", label="supercell")
    ax1.set_xticks(x[::3])
    ax1.set_xticklabels([f"{rs[i]['spec']:03d}" for i in range(0, len(rs), 3)],
                        fontsize=7)
    ax1.set_xlabel("geometry (sorted by supercell error)")
    ax1.set_ylabel(r"raw RMSE$(A)$")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.25, lw=0.6)

    gain = [r["rmse_ideal"] - r["rmse_oracle"] for r in rs]
    cols = ["#bf4520" if r["n_missing"] else "#4f5d75" for r in rs]
    xkey = "eps_eff" if any(abs(r["eps_eff"] - 1.0) > 1e-3 for r in rs) \
        else "fill_eff"
    ax2.scatter([r[xkey] for r in rs], gain, c=cols, s=18)
    ax2.axhline(0.0, color="#2d3142", lw=0.8, ls=":")
    ax2.set_xlabel(f"oracle effective {xkey.replace('_eff', '')}")
    ax2.set_ylabel(r"oracle gain over ideal  [RMSE$(A)$]")
    ax2.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(0)
    d_ideal = np.load(a.ideal_data, allow_pickle=True)
    d_super = np.load(a.super_data, allow_pickle=True)
    wl = d_ideal["wl"].astype(np.float64)
    group = d_ideal["group"].astype(np.int64)
    y_abs = d_ideal["Y_abs"].astype(np.float64)
    ideal_rt = d_ideal["RCWA_abs"].astype(np.float64)
    super_rt = d_super["RCWA_abs"].astype(np.float64)
    # both datasets share the row order / group split; assert once
    assert np.array_equal(group, d_super["group"].astype(np.int64))
    assert np.allclose(wl, d_super["wl"].astype(np.float64))

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
        miss = int(np.asarray(spec.missing, dtype=bool).sum())

        ideal_A = 1.0 - ideal_rt[m, 0] - ideal_rt[m, 1]
        super_A = 1.0 - super_rt[m, 0] - super_rt[m, 1]

        (params, full_mse_a, fit_mse, r_or, t_or), fit_idx = fit_oracle(
            spec, wl_s, Y, tin, a)
        or_A = 1.0 - r_or - t_or

        row = {
            "spec": sid,
            "n_missing": miss,
            "period0": float(spec.period),
            "amplitude0": float(spec.amplitude),
            "base0": float(spec.base_thickness),
            "period_eff": params[0],
            "amplitude_eff": params[1],
            "base_eff": params[2],
            "fill_eff": params[3],
            "eps_eff": params[4],
            "p_ratio": params[0] / float(spec.period),
            "a_ratio": params[1] / float(spec.amplitude),
            "b_ratio": params[2] / float(spec.base_thickness),
            "rmse_ideal": _rmse(ideal_A, Y[:, 2]),
            "rmse_oracle": _rmse(or_A, Y[:, 2]),
            "rmse_super": _rmse(super_A, Y[:, 2]),
            "rmse_oracle_fit": _rmse(or_A[fit_idx], Y[fit_idx, 2]),
            "full_mse_A": full_mse_a,
            "fit_mse": fit_mse,
            "fit_wl_idx": fit_idx.tolist(),
        }
        # also report R/T errors, since the fit is on (R, T)
        row["rmse_R_oracle"] = _rmse(r_or, Y[:, 0])
        row["rmse_T_oracle"] = _rmse(t_or, Y[:, 1])
        rows.append(row)
        print(f"[{k + 1}/{len(ids)}] spec {sid:03d} miss={miss} "
              f"p/p0={row['p_ratio']:.3f} a/a0={row['a_ratio']:.3f} "
              f"b/b0={row['b_ratio']:.3f} fill={row['fill_eff']:.3f} "
              f"eps={row['eps_eff']:.3f}  "
              f"A ideal/oracle/super = {row['rmse_ideal']:.4f}/"
              f"{row['rmse_oracle']:.4f}/{row['rmse_super']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    def subset(rows, pred):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return None
        out = {}
        for key in ("rmse_ideal", "rmse_oracle", "rmse_super"):
            v = [r[key] for r in sel]
            out[key] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        out["n"] = len(sel)
        out["gain_vs_ideal"] = out["rmse_ideal"]["mean"] - out["rmse_oracle"]["mean"]
        out["gap_to_super"] = out["rmse_oracle"]["mean"] - out["rmse_super"]["mean"]
        return out

    summary = {
        "all": subset(rows, lambda r: True),
        "jitter_only": subset(rows, lambda r: r["n_missing"] == 0),
        "missing_ridge": subset(rows, lambda r: r["n_missing"] > 0),
    }

    os.makedirs(a.out_dir, exist_ok=True)
    out = {"config": vars(a), "summary": summary, "per_spec": rows}
    with open(os.path.join(a.out_dir, f"{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    lines = ["# E4a — oracle effective periodic config", "",
             f"- geometries: {len(rows)}  (fit on {a.n_fit_wl} λ, "
             f"evaluated on all 16)",
             f"- family: period × amplitude × base × fill "
             f"(p {P_RANGE}, a {A_RANGE}, b {B_RANGE}, fill {F_RANGE})",
             f"- solver: ideal unit cell, nh={a.nh}, nx={a.nx}, "
             f"{a.n_layers} layers",
             f"- optimiser: Adam, {a.iters} steps × {a.restarts} restarts", "",
             "## Raw RMSE(A) by subset", "",
             "| subset | n | ideal cell | oracle config | supercell | "
             "oracle gain | gap to supercell |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    names = {"all": "all", "jitter_only": "jitter only",
             "missing_ridge": "missing ridge"}
    for key, name in names.items():
        s = summary[key]
        if s is None:
            continue
        lines.append(
            f"| {name} | {s['n']} | {s['rmse_ideal']['mean']:.4f} ± "
            f"{s['rmse_ideal']['std']:.4f} | "
            f"{s['rmse_oracle']['mean']:.4f} ± {s['rmse_oracle']['std']:.4f} | "
            f"{s['rmse_super']['mean']:.4f} ± {s['rmse_super']['std']:.4f} | "
            f"{s['gain_vs_ideal']:+.4f} | {s['gap_to_super']:+.4f} |")
    lines += ["", "## Per geometry", "",
              "| spec | miss | miss idx | p/p0 | a/a0 | b/b0 | fill | eps | ideal | oracle | super |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        sp = make_spec(r["spec"])
        mi = int(np.argmax(np.asarray(sp.missing, dtype=bool))) \
            if r["n_missing"] else -1
        lines.append(
            f"| {r['spec']:03d} | {r['n_missing']} | {mi} | "
            f"{r['p_ratio']:.3f} | "
            f"{r['a_ratio']:.3f} | {r['b_ratio']:.3f} | {r['fill_eff']:.3f} | "
            f"{r['eps_eff']:.3f} | "
            f"{r['rmse_ideal']:.4f} | {r['rmse_oracle']:.4f} | "
            f"{r['rmse_super']:.4f} |")
    lines.append("")
    path = os.path.join(a.out_dir, f"{a.tag}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    plot(rows, os.path.join(a.out_dir, f"{a.tag}.png"))

    print("\nraw RMSE(A), mean ± std")
    for key, name in names.items():
        s = summary[key]
        if s is None:
            continue
        print(f"  {name:14s} ideal {s['rmse_ideal']['mean']:.4f}  "
              f"oracle {s['rmse_oracle']['mean']:.4f}  "
              f"super {s['rmse_super']['mean']:.4f}")
    print(f"wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
