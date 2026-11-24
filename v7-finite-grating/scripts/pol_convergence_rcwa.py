"""RCWA side of the p-pol convergence / contrast study.

Reads the npz spectra written by ``scripts/pol_convergence.py`` (MEEP) and
compares them with the padded-supercell RCWA at several Fourier orders.  For
the constant-eps ablation cases the RCWA uses the same ``eps_override`` so the
comparison isolates the solver, not the material model.

Run in the torch environment:

    python scripts/pol_convergence_rcwa.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description="RCWA comparison for poly study")
    p.add_argument("--dir", default=os.path.join(_HERE, "..", "reports", "m2",
                                                 "pol_study"))
    p.add_argument("--nh", type=int, nargs="+", default=[32, 64, 96])
    p.add_argument("--padding", type=float, default=1.0)
    p.add_argument("--grid-nm", type=float, default=10.0)
    p.add_argument("--n-layers", type=int, default=40)
    return p.parse_args(argv)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main(argv=None):
    a = parse(argv)
    from m2_data import make_spec
    from rcwa_supercell import RCWAConfig, spectrum_supercell

    files = sorted(glob.glob(os.path.join(a.dir, "*.npz")))
    if not files:
        print(f"no spectra in {a.dir}; run scripts/pol_convergence.py first")
        return 1

    print(f"{'case':24s} {'nh':>4s} {'A_meep':>8s} {'A_rcwa':>8s} "
          f"{'bias':>8s} {'RMSE_A':>8s} {'RMSE_R':>8s} {'RMSE_T':>8s}")
    for f in files:
        d = np.load(f)
        spec = make_spec(int(d["spec_id"]))
        eps = float(d["eps"])
        wl = d["wl"]
        A_meep = 1 - d["R"] - d["T"]
        for nh in a.nh:
            cfg = RCWAConfig(padding_um=a.padding, nh=nh,
                             grid_nm=a.grid_nm, n_layers=a.n_layers)
            res = spectrum_supercell(
                spec, wl, cfg, pol=str(d["pol"]), angle_deg=float(d["angle"]),
                eps_override=None if eps < 0 else complex(eps, 0.0),
            )
            A = 1 - res["R"] - res["T"]
            print(f"{os.path.basename(f)[:-4]:24s} {nh:4d} {A_meep.mean():8.4f} "
                  f"{A.mean():8.4f} {A.mean() - A_meep.mean():+8.4f} "
                  f"{rmse(A, A_meep):8.4f} {rmse(res['R'], d['R']):8.4f} "
                  f"{rmse(res['T'], d['T']):8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
