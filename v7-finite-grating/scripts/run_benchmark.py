"""Run the M1 RCWA-vs-MEEP sweep and write the comparison report (torch env).

Example
-------
    python scripts/run_benchmark.py \
        --reference reports/reference/grating8_defects \
        --pol s --padding 0.5 1.0 2.0 4.0 --nh 8 16 32 64 \
        --grid-nm 10 --n-layers 60 \
        --out reports/m1_grating8_defects_s

Reads ``<reference>_<pol>.csv``/``.json`` produced by ``make_reference.py``.
"""

from __future__ import annotations

import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import argparse  # noqa: E402

from benchmark import (  # noqa: E402
    compare_against_reference,
    deembed_reference,
    load_reference,
    run_rcwa_sweep,
    write_report,
)
from rcwa_supercell import RCWAConfig  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description="M1 RCWA supercell sweep")
    p.add_argument("--reference", type=str, required=True,
                   help="reference basename or <basename>.csv/.json")
    p.add_argument("--pol", type=str, default="s", choices=["s", "p"])
    p.add_argument("--padding", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 4.0], help="supercell padding (µm)")
    p.add_argument("--nh", type=int, nargs="+", default=[8, 16, 32, 64],
                   help="Fourier orders in x")
    p.add_argument("--grid-nm", type=float, default=10.0)
    p.add_argument("--n-layers", type=int, default=60)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--out", type=str, required=True, help="output basename")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main(argv=None):
    a = parse()

    base = a.reference
    if base.endswith(".csv"):
        stem = base[:-4]
    elif base.endswith(".json"):
        stem = base[:-5]
    else:
        stem = base
    ref = load_reference(stem if stem.endswith(f"_{a.pol}") else f"{stem}_{a.pol}")
    ref_d = deembed_reference(ref)
    spec = ref["spec"]

    wl = ref["wl_um"]
    print(f"reference: {spec.n_ridges} ridges, {len(wl)} wavelengths "
          f"({wl.min():.3f}-{wl.max():.3f} µm), pol={a.pol}, "
          f"Lx={ref['Lx_um']:.2f} µm, W={ref['W_um']:.2f} µm")

    configs = [
        RCWAConfig(padding_um=pad, nh=nh, grid_nm=a.grid_nm,
                   n_layers=a.n_layers, device=a.device)
        for pad, nh in itertools.product(a.padding, a.nh)
    ]

    results = run_rcwa_sweep(
        spec, wl, configs, pol=a.pol, angle_deg=ref["angle_deg"],
        tin_dataset=a.tin_dataset, verbose=True,
    )
    rows = compare_against_reference(ref_d, results)

    best_idx = min(range(len(rows)), key=lambda i: rows[i]["rmse_A"])
    ref_d["_best_rcwa"] = results[best_idx]
    best_row = rows[best_idx]

    csv_path = a.out + ".csv"
    md_path = a.out + ".md"
    plot_path = a.out + ".png" if a.plot else None
    write_report(rows, ref_d, csv_path=csv_path, md_path=md_path,
                 plot_path=plot_path)

    print(f"\nbest: padding={best_row['padding_um']:g} µm, nh={best_row['nh']}, "
          f"RMSE(A)={best_row['rmse_A']:.4f}")
    print(f"wrote {os.path.normpath(md_path)}")
    print(f"wrote {os.path.normpath(csv_path)}")
    if plot_path:
        print(f"wrote {os.path.normpath(plot_path)}")


if __name__ == "__main__":
    main()
