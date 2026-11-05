"""Experiment 1: is the padded supercell needed, or does an ideal unit cell do?

Reuses the 24 MEEP references from ``reports/m2/data`` (via ``dataset.npz``) and
compares three estimates of the aperture response:

1. **ideal unit cell** — one clean period (``n_ridges=1``, no defects), the
   cheapest periodic solve;
2. **supercell** — the actual defective profile in a padded supercell
   (already stored in ``dataset.npz`` as the M2 RCWA input);
3. **MEEP** — the finite-grating reference.

It reports raw RMSE, stratified by missing-ridge defect, plus per-solve timing,
and optionally writes an ``dataset_ideal.npz`` so the M2 correction can be
retrained on the cheap backbone for an apples-to-apples comparison.

Runs in the TORCWA/torch environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402

from benchmark import deembed_rcwa  # noqa: E402
from grating import GratingSpec  # noqa: E402
from m2_data import FEATURE_NAMES, build_features, make_spec  # noqa: E402
from metrics import spectral_rmse  # noqa: E402
from rcwa_supercell import RCWAConfig, spectrum_supercell  # noqa: E402

KEYS = ("R", "T", "A")


def parse():
    p = argparse.ArgumentParser(description="Experiment 1: ideal unit cell vs supercell")
    p.add_argument("--data", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "dataset.npz"))
    p.add_argument("--pol", type=str, default="s", choices=["s", "p"])
    p.add_argument("--nh", type=int, default=32, help="Fourier orders (same for both)")
    p.add_argument("--grid-nm", type=float, default=10.0)
    p.add_argument("--n-layers", type=int, default=40)
    p.add_argument("--super-padding", type=float, default=1.0,
                   help="padding of the supercell baseline (matches M2)")
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--time-specs", type=int, default=5,
                   help="specs used to time the supercell solve")
    p.add_argument("--build-ideal-dataset", action="store_true")
    p.add_argument("--ideal-data-out", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "dataset_ideal.npz"))
    p.add_argument("--out", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2",
                                        "experiment1_ideal_vs_supercell"))
    return p.parse_args()


def ideal_spec_from(spec: GratingSpec) -> GratingSpec:
    """A clean single period with the same period/amplitude/base."""
    return GratingSpec(
        n_ridges=1,
        period=spec.period,
        amplitude=spec.amplitude,
        base_thickness=spec.base_thickness,
        substrate_index=spec.substrate_index,
        phase=spec.phase,
    )


def _agg(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


def main(argv=None):
    a = parse()

    d = np.load(a.data, allow_pickle=True)
    Y_abs = d["Y_abs"].astype(np.float32)          # MEEP de-embedded
    RCWA_super = d["RCWA_abs"].astype(np.float32)  # supercell de-embedded
    group = d["group"].astype(np.int64)
    wl_all = d["wl"].astype(np.float64)
    spec_ids = [int(s) for s in d["spec_ids"]]

    from tin_dispersion import get_or_fit

    tin_params = get_or_fit(
        a.tin_dataset,
        cache_path=os.path.join(_HERE, "..", "data", f"tin_fit_{a.tin_dataset}.json"),
    )

    cfg_ideal = RCWAConfig(padding_um=0.0, nh=a.nh, grid_nm=a.grid_nm,
                           n_layers=a.n_layers, device=a.device)

    rows = []
    ideal_X, ideal_RCWA, ideal_Yres, ideal_Yabs, ideal_group, ideal_wl = (
        [], [], [], [], [], []
    )
    t_ideal = []

    for idx in spec_ids:
        m = group == idx
        wl = wl_all[m]
        spec = make_spec(idx)
        ispec = ideal_spec_from(spec)

        t0 = time.time()
        res = spectrum_supercell(
            ispec, wl, cfg_ideal, pol=a.pol,
            tin_params=tin_params, tin_dataset=a.tin_dataset,
        )
        t_ideal.append(time.time() - t0)
        ideal = deembed_rcwa(res, 0.0, a.pol, spec.substrate_index)

        meep = {"R": Y_abs[m, 0], "T": Y_abs[m, 1], "A": Y_abs[m, 2]}
        superc = {"R": RCWA_super[m, 0], "T": RCWA_super[m, 1], "A": RCWA_super[m, 2]}

        row = {
            "spec": idx,
            "period_um": spec.period,
            "amplitude_um": spec.amplitude,
            "n_missing": int(spec.missing.sum()),
            "jitter_std": float(spec.ridge_scale.std()),
        }
        for k in KEYS:
            row[f"ideal_{k}"] = spectral_rmse(ideal[k], meep[k])
            row[f"super_{k}"] = spectral_rmse(superc[k], meep[k])
        rows.append(row)

        ideal_X.append(build_features(spec, wl, ideal["R"], ideal["T"], ideal["A"]))
        ideal_RCWA.append(np.column_stack([ideal["R"], ideal["T"], ideal["A"]]))
        ideal_Yres.append(np.column_stack([meep["R"] - ideal["R"],
                                           meep["T"] - ideal["T"]]))
        ideal_Yabs.append(Y_abs[m])
        ideal_group.append(np.full(len(wl), idx, dtype=np.int32))
        ideal_wl.append(wl.astype(np.float32))

        print(f"[{idx:03d}] ideal A={row['ideal_A']:.4f}  "
              f"super A={row['super_A']:.4f}  "
              f"missing={row['n_missing']}  t_ideal={t_ideal[-1]:.2f}s")

    # ---- timing of the supercell baseline on a subset ---- #
    t_super = []
    cfg_super = RCWAConfig(padding_um=a.super_padding, nh=a.nh,
                           grid_nm=a.grid_nm, n_layers=a.n_layers, device=a.device)
    for idx in spec_ids[: max(1, a.time_specs)]:
        m = group == idx
        wl = wl_all[m]
        spec = make_spec(idx)
        t0 = time.time()
        spectrum_supercell(spec, wl, cfg_super, pol=a.pol,
                           tin_params=tin_params, tin_dataset=a.tin_dataset)
        t_super.append(time.time() - t0)

    # ---- aggregate, stratified by missing-ridge defect ---- #
    def _summary(sel):
        out = {}
        for k in KEYS:
            mi, si = _agg([r[f"ideal_{k}"] for r in sel])
            ms, ss = _agg([r[f"super_{k}"] for r in sel])
            out[k] = {"ideal_mean": mi, "ideal_std": si,
                      "super_mean": ms, "super_std": ss}
        return out

    groups = {
        "all": rows,
        "no_missing": [r for r in rows if r["n_missing"] == 0],
        "missing": [r for r in rows if r["n_missing"] > 0],
    }
    summary = {name: _summary(sel) for name, sel in groups.items() if sel}

    # ---- report ---- #
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    csv_path = a.out + ".csv"
    fields = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(fields) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in fields) + "\n")

    lines = [
        "# Experiment 1 — ideal unit cell vs padded supercell",
        "",
        f"- geometries: {len(rows)} (from the M2 dataset)",
        f"- RCWA: nh={a.nh}, grid={a.grid_nm:g} nm, {a.n_layers} layers, "
        f"pol={a.pol}; supercell padding={a.super_padding:g} µm",
        "- ideal unit cell = one clean period, no defects (blind to defects)",
        "- supercell = actual defective profile, padded",
        "",
        "## Raw RCWA accuracy vs MEEP (RMSE of de-embedded spectra)",
        "",
        "| subset | n | ideal RMSE(A) | supercell RMSE(A) | ideal RMSE(R) | "
        "supercell RMSE(R) | ideal RMSE(T) | supercell RMSE(T) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "no_missing", "missing"):
        if name not in summary:
            continue
        s = summary[name]
        n = len(groups[name])
        lines.append(
            f"| {name} | {n} | "
            f"{s['A']['ideal_mean']:.4f} ± {s['A']['ideal_std']:.4f} | "
            f"{s['A']['super_mean']:.4f} ± {s['A']['super_std']:.4f} | "
            f"{s['R']['ideal_mean']:.4f} | {s['R']['super_mean']:.4f} | "
            f"{s['T']['ideal_mean']:.4f} | {s['T']['super_mean']:.4f} |"
        )

    mi = _agg(t_ideal)
    ms = _agg(t_super)
    lines += [
        "",
        "## Solve cost (seconds per spectrum of 16 wavelengths)",
        "",
        f"- ideal unit cell: {mi[0]:.2f} s (n={len(t_ideal)})",
        f"- supercell: {ms[0]:.2f} s (n={len(t_super)})",
        f"- speed-up: {ms[0] / mi[0]:.2f}×" if mi[0] > 0 else "",
        "",
        "Note: RCWA cost is dominated by the Fourier-order eigendecomposition, "
        "which depends on `nh`, not on the supercell width. The unit cell is "
        "cheaper mainly because it is under-resolved in the supercell for the "
        "same `nh`; a fair accuracy-matched comparison may need more orders for "
        "the supercell (see the M1 sweep).",
        "",
        "## Per-geometry RMSE(A)",
        "",
        "| spec | missing | jitter std | ideal | supercell |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['spec']:03d} | {r['n_missing']} | "
                     f"{r['jitter_std']:.3f} | {r['ideal_A']:.4f} | "
                     f"{r['super_A']:.4f} |")
    lines.append("")

    md_path = a.out + ".md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    with open(a.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary,
                   "timing": {"ideal": list(mi), "supercell": list(ms)},
                   "config": vars(a), "per_spec": rows}, fh, indent=2)

    print("\n=== summary ===")
    for name in ("all", "no_missing", "missing"):
        if name in summary:
            s = summary[name]["A"]
            print(f"{name:12s} n={len(groups[name]):2d}  "
                  f"ideal A={s['ideal_mean']:.4f}  super A={s['super_mean']:.4f}")
    print(f"timing: ideal {mi[0]:.2f}s  supercell {ms[0]:.2f}s  "
          f"speed-up {ms[0]/mi[0]:.2f}x")
    print(f"wrote {os.path.normpath(md_path)}")

    # ---- optional: build the ideal dataset for the M2 retrain ---- #
    if a.build_ideal_dataset:
        np.savez_compressed(
            a.ideal_data_out,
            X=np.concatenate(ideal_X),
            Y_res=np.concatenate(ideal_Yres),
            Y_abs=np.concatenate(ideal_Yabs),
            RCWA_abs=np.concatenate(ideal_RCWA),
            group=np.concatenate(ideal_group),
            wl=np.concatenate(ideal_wl),
            feature_names=np.asarray(FEATURE_NAMES),
            spec_ids=np.asarray(spec_ids, dtype=np.int32),
            rcwa_cfg=np.asarray(json.dumps({"padding_um": 0.0, "nh": a.nh,
                                            "grid_nm": a.grid_nm,
                                            "n_layers": a.n_layers})),
        )
        print(f"wrote {os.path.normpath(a.ideal_data_out)}")


if __name__ == "__main__":
    main()
