"""Run the Experiment 3 ladder and write a consolidated report.

Modes
-----
``fair``     fixed vs global vs ridge encoder at the full budget.
``scaling``  sample-efficiency curve: n_train x encoder x seed.
``e2e``      RCWA-in-the-loop config head vs a matched control.

The cheap modes are precomputed-RCWA runs (seconds each).  ``e2e`` solves RCWA
inside training and is the only expensive mode.

Examples
--------
    python scripts/run_e3_suite.py --mode fair --mode scaling
    python scripts/run_e3_suite.py --mode e2e --e2e-epochs 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import train_e3 as e3  # noqa: E402

OUT = os.path.join(_HERE, "..", "reports", "m2", "e3")
RUNS = os.path.join(OUT, "runs")


def _run(tag, **kw):
    cfg = e3.parse([])
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.quiet = True
    cfg.tag = tag
    cfg.out_dir = RUNS
    return e3.run_experiment(cfg)


def _fmt(m, key):
    return f"{m['metrics']['corr_' + key]['mean']:.4f} ± " \
           f"{m['metrics']['corr_' + key]['std']:.4f}"


def suite_fair(seeds, epochs, patience):
    rows = []
    for seed in seeds:
        for enc in ("none", "global", "ridge"):
            tag = f"fair_{enc}_s{seed}"
            t0 = time.time()
            r = _run(tag, encoder=enc, epochs=epochs, patience=patience,
                     eval_every=10, seed=seed)
            rows.append({"encoder": enc, "seed": seed,
                         "wall_s": time.time() - t0, **r})
            print(f"  {enc} s{seed}: A={_fmt(r, 'A')} "
                  f"R={_fmt(r, 'R')} params={r['n_params']}", flush=True)
    return rows


def suite_scaling(seeds, n_train, epochs, patience):
    rows = []
    for n in n_train:
        for enc in ("none", "global", "ridge"):
            for seed in seeds:
                tag = f"scale_{enc}_n{n}_s{seed}"
                r = _run(tag, encoder=enc, epochs=epochs, patience=patience,
                         eval_every=10, seed=seed, n_train=n)
                rows.append({"n_train": n, "encoder": enc, "seed": seed, **r})
                print(f"  n={n} {enc} s{seed}: A={_fmt(r, 'A')}", flush=True)
    return rows


def suite_e2e(seeds, cfg_modes, epochs, batch_size):
    rows = []
    for seed in seeds:
        for enc in ("global", "ridge"):
            for mode in cfg_modes:
                tag = f"e2e_{enc}_{mode}_s{seed}"
                t0 = time.time()
                r = _run(tag, encoder=enc, cfg_mode=mode, in_loop=True,
                         epochs=epochs, batch_size=batch_size, eval_every=2,
                         patience=max(4, epochs // 3), seed=seed)
                rows.append({"encoder": enc, "cfg_mode": mode, "seed": seed,
                             "wall_s": time.time() - t0, **r})
                print(f"  e2e {enc}/{mode} s{seed}: A={_fmt(r, 'A')} "
                      f"({time.time() - t0:.0f}s)", flush=True)

            # matched control: same data exposure and schedule, precomputed
            # RCWA (so the only difference is the differentiable config path)
            tag = f"control_{enc}_s{seed}"
            r = _run(tag, encoder=enc, epochs=epochs, batch_size=batch_size,
                     eval_every=2, patience=max(4, epochs // 3), seed=seed)
            rows.append({"encoder": enc, "cfg_mode": "none", "seed": seed,
                         "control": True, **r})
            print(f"  control {enc} s{seed}: A={_fmt(r, 'A')}", flush=True)
    return rows


def _merge(old, new, keys):
    """Replace rows in ``old`` that share a key with rows from ``new``."""
    seen = {tuple(r[k] for k in keys) for r in new}
    return [r for r in old if tuple(r[k] for k in keys) not in seen] + new


def _agg(rows, group_key, metric="A"):
    out = {}
    for enc in sorted({r[group_key] for r in rows}):
        vals = [r["metrics"][f"corr_{metric}"]["mean"]
                for r in rows if r[group_key] == enc]
        out[enc] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def write_report(fair, scaling, e2e_rows):
    lines = ["# Experiment 3 — fair encoder comparison and redesign", ""]

    if fair:
        lines += [
            "## Fair ladder (same data, loss, optimiser and schedule)", "",
            "All arms: ideal unit cell `nh=8`, precomputed RCWA, 18/6 geometry "
            "split, MSE on standardised `(dR, dT)`, batch 64, AdamW "
            "lr=1e-3 wd=1e-3.", "",
            "| encoder | RMSE R | RMSE T | RMSE A | params | wall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for enc in ("none", "global", "ridge"):
            sel = [r for r in fair if r["encoder"] == enc]
            if not sel:
                continue
            def mean_std(key):
                v = [r["metrics"][f"corr_{key}"]["mean"] for r in sel]
                return f"{np.mean(v):.4f} ± {np.std(v):.4f}"
            lines.append(
                f"| {enc} | {mean_std('R')} | {mean_std('T')} | "
                f"{mean_std('A')} | {sel[0]['n_params']} | "
                f"{np.mean([r['wall_s'] for r in sel]):.1f}s |")
        lines.append("")

    if scaling:
        lines += [
            "## Sample efficiency (held-out RMSE(A), mean ± std over seeds)", "",
            "| n_train | fixed | global | ridge |", "|---:|---:|---:|---:|",
        ]
        ns = sorted({r["n_train"] for r in scaling})
        for n in ns:
            cells = []
            for enc in ("none", "global", "ridge"):
                v = [r["metrics"]["corr_A"]["mean"] for r in scaling
                     if r["n_train"] == n and r["encoder"] == enc]
                cells.append(f"{np.mean(v):.4f} ± {np.std(v):.4f}" if v else "—")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")

    if e2e_rows:
        lines += [
            "## Differentiable config head (matched data exposure)", "",
            "| encoder | cfg | RMSE A | train wall |", "|---|---|---:|---:|",
        ]
        for r in e2e_rows:
            lines.append(
                f"| {r['encoder']} | {r['cfg_mode']} | "
                f"{r['metrics']['corr_A']['mean']:.4f} | "
                f"{r['wall_s']:.0f}s |")
        lines.append("")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "e3_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.normpath(path)}")
    if scaling:
        _plot_scaling(scaling)


def _plot_scaling(rows):
    """Held-out RMSE(A) vs number of training geometries, mean ± std."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    ns = sorted({r["n_train"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=160)
    style = {
        "none": ("#2d3142", "o", "fixed geometry"),
        "global": ("#bf4520", "s", "global encoder"),
        "ridge": ("#2e5aa8", "^", "ridge encoder"),
    }
    for enc in ("none", "global", "ridge"):
        color, marker, label = style[enc]
        dx = {"none": -0.06, "global": 0.0, "ridge": 0.06}[enc]
        ys, es = [], []
        for n in ns:
            v = [r["metrics"]["corr_A"]["mean"] for r in rows
                 if r["n_train"] == n and r["encoder"] == enc]
            ys.append(float(np.mean(v)) if v else np.nan)
            es.append(float(np.std(v)) if v else 0.0)
        ax.errorbar(np.asarray(ns) + dx, ys, yerr=es, marker=marker, ms=4.5,
                    lw=1.4, capsize=3, color=color, label=label)
    ax.set_xlabel("training geometries")
    ax.set_ylabel(r"held-out RMSE$(A)$")
    ax.set_xticks(ns)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("mean over seeds; bars = seed std", fontsize=8,
                 color="#4f5d75")
    fig.tight_layout()
    out = os.path.join(OUT, "e3_scaling.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {os.path.normpath(out)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Experiment 3 suites")
    p.add_argument("--mode", action="append",
                   choices=["fair", "scaling", "e2e", "report"], default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-train", type=int, nargs="+", default=[6, 12, 18])
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--patience", type=int, default=150)
    p.add_argument("--e2e-epochs", type=int, default=24)
    p.add_argument("--e2e-batch", type=int, default=64)
    p.add_argument("--e2e-cfg-modes", nargs="+", default=["amp", "amp_period"])
    p.add_argument("--e2e-seeds", type=int, nargs="+", default=[0])
    a = p.parse_args(argv)
    modes = a.mode or ["fair", "scaling"]

    # merge with a previous suite run so re-running one mode keeps the others
    path = os.path.join(OUT, "e3_suite.json")
    legacy = os.path.join(_HERE, "..", "reports", "m2", "e3_suite.json")
    prev = {}
    for candidate in (path, legacy):
        if os.path.exists(candidate):
            with open(candidate) as fh:
                prev = json.load(fh)
            break

    if modes == ["report"]:
        write_report(prev.get("fair", []), prev.get("scaling", []),
                     prev.get("e2e", []))
        return

    fair, scaling, e2e_rows = [], [], []
    if "fair" in modes:
        print("== fair ladder ==", flush=True)
        fair = suite_fair(a.seeds, a.epochs, a.patience)
        fair = _merge(prev.get("fair", []), fair, ("encoder", "seed"))
    else:
        fair = prev.get("fair", [])
    if "scaling" in modes:
        print("== scaling ==", flush=True)
        scaling = suite_scaling(a.seeds, a.n_train, a.epochs, a.patience)
        scaling = _merge(prev.get("scaling", []), scaling,
                         ("n_train", "encoder", "seed"))
    else:
        scaling = prev.get("scaling", [])
    if "e2e" in modes:
        print("== e2e ==", flush=True)
        e2e_rows = suite_e2e(a.e2e_seeds, a.e2e_cfg_modes,
                             a.e2e_epochs, a.e2e_batch)
    else:
        e2e_rows = prev.get("e2e", [])

    write_report(fair, scaling, e2e_rows)
    with open(path, "w") as fh:
        json.dump({"fair": fair, "scaling": scaling, "e2e": e2e_rows}, fh,
                  indent=2)
    print("wrote reports/m2/e3_suite.json")


if __name__ == "__main__":
    main()
