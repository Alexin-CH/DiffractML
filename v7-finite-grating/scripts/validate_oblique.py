"""Validate the oblique de-embedding against a MEEP reference.

The de-embedding ``A=(1-R-T)·Lx/W`` and the bare-substrate baseline are
angle-aware through ``bare_substrate(angle, pol, n_sub)``.  This script runs the
padded supercell RCWA at the same angle/polarisation as a reference and reports
the residual error, before any oblique reference is used for training.

Example
-------
    python scripts/validate_oblique.py --spec 3 --pol s --angle 30
    python scripts/validate_oblique.py --spec 3 --pol p --angle 60 --nh 16
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402

from benchmark import deembed_reference, deembed_rcwa, load_reference  # noqa: E402
from m2_data import reference_base  # noqa: E402
from rcwa_supercell import RCWAConfig, spectrum_supercell  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description="Validate oblique de-embedding")
    p.add_argument("--spec", type=int, required=True)
    p.add_argument("--pol", choices=["s", "p"], required=True)
    p.add_argument("--angle", type=float, required=True)
    p.add_argument("--data-dir", default=os.path.join(
        _HERE, "..", "reports", "m2", "data140"))
    p.add_argument("--spec-set", choices=["m2", "resonant"], default="m2")
    p.add_argument("--nh", type=int, default=32)
    p.add_argument("--padding", type=float, default=1.0)
    p.add_argument("--grid-nm", type=float, default=10.0)
    p.add_argument("--n-layers", type=int, default=40)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-rmse-a", type=float, default=0.03,
                   help="fail above this absorption RMSE")
    return p.parse_args(argv)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def anomalies(period, angle_deg, n_sub, wl_lo, wl_hi):
    """Rayleigh wavelengths for m=1 orders in air and substrate."""
    th = np.deg2rad(angle_deg)
    out = {}
    for medium, n in (("air", 1.0), ("sub", n_sub)):
        for sign, name in ((+1, "+1"), (-1, "-1")):
            lam = n * period * (1.0 + sign * np.sin(th))
            if wl_lo <= lam <= wl_hi:
                out[f"{medium} m={name}"] = float(lam)
    return out


def main(argv=None):
    a = parse(argv)
    base = os.path.join(a.data_dir,
                        reference_base(a.spec, a.pol, a.angle, a.spec_set))
    ref = load_reference(base)
    ref_d = deembed_reference(ref)
    spec = ref["spec"]
    wl = ref_d["wl_um"]

    cfg = RCWAConfig(padding_um=a.padding, nh=a.nh, grid_nm=a.grid_nm,
                     n_layers=a.n_layers, device=a.device)
    res = spectrum_supercell(spec, wl, cfg, pol=a.pol,
                             angle_deg=ref["angle_deg"],
                             tin_dataset="beliaev")
    rcwa_d = deembed_rcwa(res, ref["angle_deg"], a.pol,
                          spec.substrate_index)

    rows = []
    for key in ("R", "T", "A"):
        rows.append((key, _rmse(rcwa_d[key], ref_d[key])))
    energy_ref = float(np.max(np.abs(ref_d["R"] + ref_d["T"] + ref_d["A"] - 1)))
    energy_rcwa = float(np.max(np.abs(rcwa_d["R"] + rcwa_d["T"]
                                      + rcwa_d["A"] - 1)))

    print(f"spec {a.spec:03d}  pol={a.pol}  angle={ref['angle_deg']:g}deg  "
          f"nh={a.nh} padding={a.padding:g}")
    print(f"  wavelengths: {len(wl)} over {wl.min():.3f}-{wl.max():.3f} um")
    for key, value in rows:
        print(f"  RMSE {key}: {value:.4f}")
    print(f"  max |R+T+A-1|: meep {energy_ref:.2e}  rcwa {energy_rcwa:.2e}")
    print("  Rayleigh anomalies in band:",
          anomalies(spec.period, ref["angle_deg"], spec.substrate_index,
                    wl.min(), wl.max()) or "none")

    rmse_a = dict(rows)["A"]
    ok = rmse_a <= a.max_rmse_a and energy_rcwa < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'} (RMSE A <= {a.max_rmse_a})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
