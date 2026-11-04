"""Generate a MEEP FDTD reference for the M1 benchmark (MEEP environment).

Example
-------
    python scripts/make_reference.py \
        --n-ridges 8 --period 1.2 --amplitude 0.05 --base-thickness 0.03 \
        --jitter-std 0.12 --n-missing 1 --seed 0 \
        --wl-range 0.6 1.3 --n-freq 40 --resolution 150 \
        --out reports/reference/grating8_defects

Writes ``<out>_s.csv``/``.json`` and ``<out>_p.csv``/``.json``.  Use a high
``--resolution`` (>= ~150 px/µm) to resolve the ~30-50 nm TiN features; thin
films are under-resolved at lower values and bias R upward.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import argparse  # noqa: E402

from benchmark import add_spec_args, save_reference, spec_from_args  # noqa: E402
from meep_grating import run_grating  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description="MEEP finite-grating reference")
    add_spec_args(p)
    p.add_argument("--wl-range", type=float, nargs=2, default=[0.6, 1.3], help="µm")
    p.add_argument("--n-freq", type=int, default=40)
    p.add_argument("--pol", nargs="+", default=["s", "p"], choices=["s", "p"])
    p.add_argument("--angle", type=float, default=0.0, help="incidence angle (deg)")
    p.add_argument("--resolution", type=float, default=150.0, help="px/µm")
    p.add_argument("--dpml", type=float, default=0.5)
    p.add_argument("--pad", type=float, default=1.5, help="air pad to PML (µm)")
    p.add_argument("--air-above", type=float, default=1.0)
    p.add_argument("--sub-depth", type=float, default=0.8)
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--decay-by", type=float, default=1e-7)
    p.add_argument("--out", type=str, required=True,
                   help="output basename (no extension)")
    return p.parse_args()


def main(argv=None):
    a = parse()
    spec = spec_from_args(a)
    print(f"grating: {spec.n_ridges} ridges, period={spec.period} µm, "
          f"amp={spec.amplitude} µm, base={spec.base_thickness} µm")
    print(f"defects: scale std={float(__import__('numpy').std(spec.ridge_scale)):.3f}, "
          f"missing={int(spec.missing.sum())}")

    for pol in a.pol:
        print(f"[MEEP] pol={pol}, angle={a.angle}°, res={a.resolution} px/µm ...")
        result = run_grating(
            spec,
            wl_range=tuple(a.wl_range),
            n_freq=a.n_freq,
            angle_deg=a.angle,
            pol=pol,
            resolution=a.resolution,
            dpml=a.dpml,
            pad=a.pad,
            air_above=a.air_above,
            sub_depth=a.sub_depth,
            tin_dataset=a.tin_dataset,
            decay_by=a.decay_by,
            verbose=True,
        )
        csv_path, json_path = save_reference(
            result, f"{a.out}_{pol}", spec=spec,
            extra={"n_freq": a.n_freq, "wl_range": list(a.wl_range)},
        )
        print(f"  saved {os.path.normpath(csv_path)}")
        print(f"  saved {os.path.normpath(json_path)}")


if __name__ == "__main__":
    main()
