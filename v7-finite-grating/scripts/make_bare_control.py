"""Measure the bare-substrate instrument response of a campaign cell.

At oblique incidence the finite-cell reference carries a few-percent bias: the
truncated plane-wave source loses part of its beam to the side PML, and the
loss is not the same for the incident beam (travelling at the incidence angle)
and the transmitted beam (travelling at the refracted angle).  No incident-flux
calibration can reconcile the two, and widening the cell does not help because
the source widens with it.

The bias is a property of the cell, so it can be measured once per
(angle, polarisation, cell) with a bare-substrate run and divided out of every
structure reference:

    R_corr = R0_fresnel * R_struct / R_bare
    T_corr = T0_fresnel * T_struct / T_bare

The incident flux cancels completely, and every geometric bias affects the
structure and the control run identically.  ``build_m2_dataset.py`` applies the
correction (``--bare-dir``); a lossless structure must then satisfy R + T = 1,
which is what ``scripts/validate_ratio_scheme.py`` checks before a long
campaign is launched.

Example
-------
    python scripts/make_bare_control.py --angle 60 --pol s --sx 14.78 \
        --n-freq 6 --out-dir reports/m2/bare
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description="Bare-substrate control run")
    p.add_argument("--angle", type=float, required=True)
    p.add_argument("--pol", choices=["s", "p"], required=True)
    p.add_argument("--sx", type=float, required=True,
                   help="target cell width (um); the bare aperture is sized to "
                        "reproduce it")
    p.add_argument("--pad", type=float, default=3.0)
    p.add_argument("--dpml", type=float, default=0.5)
    p.add_argument("--resolution", type=float, default=140.0)
    p.add_argument("--wl-range", type=float, nargs=2, default=[0.7, 1.15])
    p.add_argument("--n-freq", type=int, default=6,
                   help="calibration wavelengths; interpolated in the builder")
    p.add_argument("--out-dir", default=os.path.join(_HERE, "..", "reports",
                                                     "m2", "bare"))
    p.add_argument("--energy-tol", type=float, default=0.20,
                   help="the control deliberately carries the cell bias")
    return p.parse_args(argv)


def control_base(angle: float, pol: str, sx: float) -> str:
    """Filename stem shared by the control writer and the dataset builder."""
    return f"bare_a{angle:g}_{pol}_sx{sx:.2f}"


def main(argv=None):
    a = parse(argv)
    from benchmark import save_reference
    from grating import GratingSpec
    from meep_grating import run_grating

    # Zero-height ridges over an aperture sized to reproduce the campaign cell:
    # sx = W + 2*pad + 2*dpml.
    w_needed = a.sx - 2.0 * (a.pad + a.dpml)
    if w_needed <= 0.0:
        raise SystemExit(f"--sx {a.sx} too small for pad {a.pad}, dpml {a.dpml}")
    spec = GratingSpec(n_ridges=1, period=w_needed, amplitude=0.0,
                       base_thickness=0.0)

    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, control_base(a.angle, a.pol, a.sx))
    print(f"bare control: angle={a.angle:g} pol={a.pol} "
          f"sx={a.sx:.3f} (aperture {w_needed:.3f}) "
          f"n_wl={a.n_freq} res={a.resolution}", flush=True)

    result = run_grating(
        spec, wl_range=tuple(a.wl_range), n_freq=a.n_freq, pol=a.pol,
        angle_deg=a.angle, resolution=a.resolution, pad=a.pad, dpml=a.dpml,
    )
    save_reference(result, base, spec=spec, energy_tol=a.energy_tol,
                   extra={"control": "bare_substrate",
                          "common_sx": a.sx,
                          "n_freq": a.n_freq,
                          "wl_range": list(a.wl_range)})
    A = 1.0 - result.R - result.T
    print(f"  saved {base}.csv  A mean={float(np.mean(A)):+.4f} "
          f"(should be 0; the departure measures the cell bias)", flush=True)


if __name__ == "__main__":
    main()
