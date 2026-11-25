"""Gate the oblique ratio correction on a lossless structure.

A lossless structure must satisfy ``R + T = 1`` exactly.  Every cell-level bias
(source near field, beam walk, side-PML clipping) violates that identity, so a
lossless grating is a self-contained gate: no second solver or analytic
reference is needed to decide whether the oblique cell is trustworthy.

Two normalisations are compared in the same cell:

* ``raw``       — per-plane incident-flux calibration (what E5 used),
* ``corrected`` — ``R = R0 * R/R_bare`` and ``T = T0 * T/T_bare``.

Example
-------
    python scripts/validate_ratio_scheme.py --angle 60 --sx 14.78 --pad 3 \
        --resolution 100 --wl 0.75 0.90 1.05
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description="Lossless R+T gate")
    p.add_argument("--spec", type=int, default=3)
    p.add_argument("--angle", type=float, default=60.0)
    p.add_argument("--pols", nargs="+", default=["s", "p"], choices=["s", "p"])
    p.add_argument("--sx", type=float, default=None,
                   help="common cell width (default: --pad with the spec's own "
                        "aperture)")
    p.add_argument("--pad", type=float, default=3.0)
    p.add_argument("--dpml", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=4.0,
                   help="lossless permittivity of the ridges")
    p.add_argument("--resolution", type=float, default=100.0)
    p.add_argument("--wl", type=float, nargs="+", default=[0.75, 0.90, 1.05])
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--out", default=os.path.join(_HERE, "..", "reports", "m2",
                                                 "ratio_gate.json"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    import meep

    from benchmark import bare_substrate
    from grating import GratingSpec
    from m2_data import make_spec
    from meep_grating import run_grating

    spec = make_spec(a.spec)
    pad = a.pad
    if a.sx is not None:
        pad = 0.5 * (a.sx - spec.aperture_width - 2.0 * a.dpml)
    sx = spec.aperture_width + 2.0 * (pad + a.dpml)
    wl_lo, wl_hi = min(a.wl), max(a.wl)
    grid = np.linspace(wl_lo, wl_hi, len(a.wl))

    print(f"lossless gate: spec={a.spec} angle={a.angle:g} sx={sx:.3f} "
          f"pad={pad:.3f} eps={a.eps:g} res={a.resolution:g}", flush=True)

    bare_spec = GratingSpec(n_ridges=1,
                            period=sx - 2.0 * (pad + a.dpml),
                            amplitude=0.0, base_thickness=0.0)
    lossless = meep.Medium(index=math.sqrt(a.eps))

    rows, ok = [], True
    for pol in a.pols:
        # Both runs use the same cell, source, monitors and wavelength grid.
        rb = run_grating(bare_spec, wl_range=(wl_lo, wl_hi), n_freq=len(a.wl),
                         pol=pol, angle_deg=a.angle, resolution=a.resolution,
                         pad=pad, dpml=a.dpml)
        rs = run_grating(spec, wl_range=(wl_lo, wl_hi), n_freq=len(a.wl),
                         pol=pol, angle_deg=a.angle, resolution=a.resolution,
                         pad=pad, dpml=a.dpml, tin_material=lossless)
        R0, T0 = bare_substrate(a.angle, pol, spec.substrate_index)
        Rc = R0 * rs.R / np.maximum(rb.R, 1e-3)
        Tc = T0 * rs.T / np.maximum(rb.T, 1e-3)
        raw = 1.0 - rs.R - rs.T
        cor = 1.0 - Rc - Tc
        print(f"  {pol}: wl={np.array2string(grid, precision=2)}", flush=True)
        print(f"     raw   R+T = {np.array2string(rs.R + rs.T, precision=4)} "
              f"(A err {np.max(np.abs(raw)):.4f})", flush=True)
        print(f"     ratio R+T = {np.array2string(Rc + Tc, precision=4)} "
              f"(A err {np.max(np.abs(cor)):.4f})", flush=True)
        for j, wl in enumerate(grid):
            rows.append({"pol": pol, "wl": float(wl),
                         "R_raw": float(rs.R[j]), "T_raw": float(rs.T[j]),
                         "R_bare": float(rb.R[j]), "T_bare": float(rb.T[j]),
                         "R_corr": float(Rc[j]), "T_corr": float(Tc[j]),
                         "A_raw": float(raw[j]), "A_corr": float(cor[j])})
        if np.max(np.abs(cor)) > a.tol:
            ok = False

    verdict = "PASS" if ok else "FAIL"
    print(f"RATIO_GATE {verdict} (tol {a.tol} on max|R+T-1| after correction)",
          flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"pass": ok, "tol": a.tol, "angle": a.angle, "sx": sx,
                   "pad": pad, "eps": a.eps, "spec": a.spec, "rows": rows},
                  fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
