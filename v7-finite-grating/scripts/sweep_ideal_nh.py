"""Follow-up to Experiment 1: can the ideal unit cell trade fewer orders for speed?

Experiment 1 showed the ideal unit cell and the padded supercell cost the same at
equal `nh`, because RCWA cost is dominated by the Fourier-order
eigendecomposition, not the supercell width.  This script asks the useful
follow-up: is the ideal cell still accurate at *fewer* orders (and therefore
actually faster), while the supercell needs more orders to resolve its ridges?

Runs in the TORCWA/torch environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402

from benchmark import deembed_rcwa  # noqa: E402
from m2_data import make_spec  # noqa: E402
from metrics import spectral_rmse  # noqa: E402
from rcwa_supercell import RCWAConfig, spectrum_supercell  # noqa: E402
from benchmark_ideal_vs_supercell import ideal_spec_from  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description="Ideal-unit-cell nh vs cost")
    p.add_argument("--data", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "dataset.npz"))
    p.add_argument("--pol", type=str, default="s", choices=["s", "p"])
    p.add_argument("--nhs", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--grid-nm", type=float, default=10.0)
    p.add_argument("--n-layers", type=int, default=40)
    p.add_argument("--n-specs", type=int, default=12, help="subset of specs to time")
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2",
                                        "experiment1_nh_cost"))
    return p.parse_args()


def main(argv=None):
    a = parse()
    d = np.load(a.data, allow_pickle=True)
    Y_abs = d["Y_abs"].astype(np.float32)
    group = d["group"].astype(np.int64)
    wl_all = d["wl"].astype(np.float64)
    spec_ids = [int(s) for s in d["spec_ids"]][: a.n_specs]

    from tin_dispersion import get_or_fit

    tin_params = get_or_fit(
        a.tin_dataset,
        cache_path=os.path.join(_HERE, "..", "data", f"tin_fit_{a.tin_dataset}.json"),
    )

    rows = []
    for nh in a.nhs:
        cfg = RCWAConfig(padding_um=0.0, nh=nh, grid_nm=a.grid_nm,
                         n_layers=a.n_layers, device=a.device)
        rmses, times = [], []
        for idx in spec_ids:
            m = group == idx
            wl = wl_all[m]
            spec = make_spec(idx)
            t0 = time.time()
            res = spectrum_supercell(
                ideal_spec_from(spec), wl, cfg, pol=a.pol,
                tin_params=tin_params, tin_dataset=a.tin_dataset,
            )
            times.append(time.time() - t0)
            ideal = deembed_rcwa(res, 0.0, a.pol, spec.substrate_index)
            meep = {"R": Y_abs[m, 0], "T": Y_abs[m, 1], "A": Y_abs[m, 2]}
            rmses.append(spectral_rmse(ideal["A"], meep["A"]))
        rows.append({
            "nh": nh,
            "rmse_A_mean": float(np.mean(rmses)),
            "rmse_A_std": float(np.std(rmses)),
            "time_s_mean": float(np.mean(times)),
        })
        print(f"nh={nh:3d}  ideal RMSE(A)={rows[-1]['rmse_A_mean']:.4f}"
              f"±{rows[-1]['rmse_A_std']:.4f}  "
              f"time={rows[-1]['time_s_mean']:.2f}s/spectrum")

    lines = [
        "# Experiment 1 follow-up — ideal unit-cell nh vs cost",
        "",
        f"- specs: {len(spec_ids)} subset of the M2 set; pol={a.pol}",
        f"- ideal unit cell, grid={a.grid_nm:g} nm, {a.n_layers} layers",
        "- reference: MEEP finite grating, de-embedded absorption",
        "",
        "| nh | ideal RMSE(A) | time (s / 16-λ spectrum) |",
        "|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['nh']} | {r['rmse_A_mean']:.4f} ± "
                     f"{r['rmse_A_std']:.4f} | {r['time_s_mean']:.2f} |")
    lines += [
        "",
        "For comparison (Experiment 1, same 16-λ spectra): supercell at nh=32 "
        "takes ~13.5 s per spectrum, so the unit cell is not cheaper at equal "
        "nh.  If a lower nh preserves accuracy here, the ideal backbone does "
        "give a genuine speed win.",
        "",
    ]
    md = a.out + ".md"
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {os.path.normpath(md)}")


if __name__ == "__main__":
    main()
