"""Gate an oblique MEEP cell against the analytic Fresnel interface.

Before spending days of FDTD time, check that the configured cell (``pad``,
``dpml``, ``air_above``) reproduces the bare-substrate R and T of an air/n
interface to within ``--tol``.  A truncated plane wave at oblique incidence
loses part of its beam to the side PML, which biases R *and* T; widening the
cell fixes it, but only up to the point where a residual few-percent
systematic remains.  Picking the pad from this measurement (rather than a
guess) is what makes the oblique reference trustworthy.

Prints one ``SELFCHECK`` line per (angle, pad, pol) and writes a JSON summary
with the smallest passing pad per angle.

Example
-------
    python scripts/flux_selfcheck.py --angles 30 60 --pads 1.5 3 4 6 \
        --pols s p --resolution 140 --out reports/m2/selfcheck.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description="Fresnel self-check for oblique cells")
    p.add_argument("--angles", type=float, nargs="+", default=[30.0, 60.0])
    p.add_argument("--pads", type=float, nargs="+", default=[1.5, 3.0, 4.0, 6.0])
    p.add_argument("--pols", nargs="+", default=["s", "p"], choices=["s", "p"])
    p.add_argument("--wl", type=float, nargs="+", default=[0.75, 1.10])
    p.add_argument("--resolution", type=float, default=140.0)
    p.add_argument("--n-sub", type=float, default=1.5)
    p.add_argument("--tol", type=float, default=0.02,
                   help="target max |R-R0| or |T-T0|; below this the pad is "
                        "preferred")
    p.add_argument("--hard-tol", type=float, default=0.05,
                   help="accept the best pad below this error even if it misses "
                        "--tol (residual side-loss bias)")
    p.add_argument("--out", default=os.path.join(
        _HERE, "..", "reports", "m2", "selfcheck.json"))
    p.add_argument("--merge", action="store_true",
                   help="append to an existing --out summary")
    return p.parse_args(argv)


def bare_spec():
    """Zero-height ridges: geometry is the substrate slab only."""
    from grating import GratingSpec

    return GratingSpec(n_ridges=1, period=1.5, amplitude=0.0, base_thickness=0.0)


def check_point(angle, pad, pol, wl, resolution, n_sub):
    from benchmark import bare_substrate
    from meep_grating import run_grating

    res = run_grating(
        bare_spec(), wl_range=(wl * 0.995, wl * 1.005), n_freq=1, pol=pol,
        angle_deg=angle, resolution=resolution, pad=pad,
    )
    R = float(np.atleast_1d(res.R)[0])
    T = float(np.atleast_1d(res.T)[0])
    R0, T0 = bare_substrate(angle, pol, n_sub)
    return {"wl": wl, "R": R, "T": T, "R0": R0, "T0": T0,
            "err": max(abs(R - R0), abs(T - T0)), "R_plus_T": R + T}


def main(argv=None):
    a = parse(argv)
    results = []
    print(f"Fresnel self-check: res={a.resolution}  tol={a.tol}  "
          f"wl={a.wl}", flush=True)
    for angle in a.angles:
        for pad in a.pads:
            for pol in a.pols:
                pts = [check_point(angle, pad, pol, wl, a.resolution, a.n_sub)
                       for wl in a.wl]
                err = max(p["err"] for p in pts)
                rpt = [p["R_plus_T"] for p in pts]
                ok = err <= a.tol
                print(f"SELFCHECK angle={angle:g} pad={pad:g} pol={pol} "
                      f"max_err={err:.4f} R+T={min(rpt):.3f}-{max(rpt):.3f} "
                      f"{'PASS' if ok else 'FAIL'}", flush=True)
                results.append({"angle": angle, "pad": pad, "pol": pol,
                                "max_err": err, "pass": ok, "points": pts})

    if a.merge and os.path.exists(a.out):
        with open(a.out) as fh:
            old = json.load(fh)
        seen = {(r["angle"], r["pad"], r["pol"]) for r in results}
        merged = [r for r in old.get("results", [])
                  if (r["angle"], r["pad"], r["pol"]) not in seen]
        results = merged + results
        print(f"merged {len(merged)} earlier result(s) from {a.out}", flush=True)

    # Aggregate per (angle, pad) over polarisations: a pad is only usable if it
    # reproduces Fresnel for both s and p.
    chosen, pad_errors = {}, {}
    for angle in a.angles:
        by_pad = {}
        for r in results:
            if r["angle"] != angle:
                continue
            key = r["pad"]
            by_pad[key] = max(by_pad.get(key, 0.0), r["max_err"])
        pad_errors[str(int(angle))] = by_pad
        passing = sorted(p for p, e in by_pad.items() if e <= a.tol)
        if passing:
            chosen[str(int(angle))] = passing[0]
            print(f"CHOSEN_PAD angle={angle:g} pad={passing[0]} "
                  f"err={by_pad[passing[0]]:.4f}", flush=True)
            continue
        # Nothing meets the target: fall back to the best pad, but only if its
        # error is small enough to be a bias we can live with (or correct for).
        best = min(by_pad, key=by_pad.get) if by_pad else None
        best_err = by_pad.get(best, float("inf"))
        if best is not None and best_err <= a.hard_tol:
            chosen[str(int(angle))] = best
            print(f"CHOSEN_PAD angle={angle:g} pad={best} err={best_err:.4f} "
                  f"WARN above tol {a.tol} but under hard ceiling {a.hard_tol}",
                  flush=True)
        else:
            chosen[str(int(angle))] = None
            print(f"CHOSEN_PAD angle={angle:g} pad=None best={best} "
                  f"err={best_err:.4f} > hard ceiling {a.hard_tol}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"tolerance": a.tol, "hard_tolerance": a.hard_tol,
                   "chosen_pad": chosen, "pad_errors": pad_errors,
                   "results": results}, fh, indent=2)

    all_ok = all(v is not None for v in chosen.values())
    print(f"SELFCHECK {'OK' if all_ok else 'INCOMPLETE'} -> {a.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
