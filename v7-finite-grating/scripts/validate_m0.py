"""M0 validation: verify the FDTD ground-truth simulator before trusting it.

Checks, in order:
  1. TiN Drude-Lorentz fit reproduces the tabulated n/k and the MEEP
     ``Medium.epsilon(f)`` matches the Python model.
  2. A planar interface reproduces the analytic Fresnel reflectance (s and p,
     normal and oblique).
  3. A lossless finite grating conserves energy (R + T = 1) and stays bounded.
  4. A real TiN finite grating with defects runs and produces a spectrum.

Run inside the MEEP environment:
    python scripts/validate_m0.py [--resolution 40] [--dataset labo]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

from fresnel import fresnel_RT  # noqa: E402
from grating import GratingSpec  # noqa: E402
from meep_grating import run_grating  # noqa: E402


def _section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_tin(dataset, wl_range, resolution):
    _section(f"1. TiN dispersion fit + MEEP medium ({dataset})")
    from tin_dispersion import (  # noqa: PLC0415
        build_meep_medium,
        epsilon_model,
        get_or_fit,
        load_nk,
        verify_medium,
    )
    import meep  # noqa: PLC0415

    cache = os.path.join(_HERE, "..", "data", f"tin_fit_{dataset}.json")
    params = get_or_fit(dataset, cache_path=cache, n_osc=4, wl_range=wl_range)
    fit = params["fit"]
    print(f"  fit band      : {fit['wl_range_um'][0]:.3f}-{fit['wl_range_um'][1]:.3f} µm")
    print(f"  n rel RMSE    : {fit['n_rel_rmse']:.4f}")
    print(f"  k rel RMSE    : {fit['k_rel_rmse']:.4f}")

    report = verify_medium(params, meep)
    print(f"  MEEP vs model : max |d eps|/|eps| = {report['max_rel']:.3e}")
    ok = report["max_rel"] < 5e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok, params


def check_fresnel(resolution):
    _section("2. Planar interface vs analytic Fresnel")
    planar = GratingSpec(
        n_ridges=1, period=1.0, amplitude=0.0, base_thickness=0.0,
        substrate_index=1.5,
    )
    dummy = __import__("meep").Medium(index=1.0)
    all_ok = True
    for pol in ("s", "p"):
        for angle in (0.0, 30.0):
            is_oblique = angle != 0.0
            n_freq = 6 if not is_oblique else 1
            wl_range = (0.6, 1.0) if not is_oblique else (0.85, 0.85)
            res = run_grating(
                planar, wl_range=wl_range, n_freq=n_freq, angle_deg=angle,
                pol=pol, resolution=resolution, tin_material=dummy,
                decay_by=1e-8, pad=2.0 if is_oblique else 0.6,
                air_above=1.2 if is_oblique else 0.6,
            )
            R_ref, T_ref = fresnel_RT(1.0, 1.5, angle, pol)
            R_meep = np.asarray(res.R)
            err = float(np.mean(np.abs(R_meep - R_ref)))
            tol = 0.02 if not is_oblique else 0.01
            ok = err < tol
            all_ok &= ok
            print(f"  pol={pol} angle={angle:4.1f}°  R_meep={R_meep.mean():.4f}  "
                  f"R_ref={R_ref:.4f}  err={err:.4f}  -> {'PASS' if ok else 'FAIL'}")
    return all_ok


def check_energy(resolution):
    _section("3. Lossless finite grating energy conservation")
    import meep  # noqa: PLC0415

    spec = GratingSpec(
        n_ridges=4, period=1.0, amplitude=0.05, base_thickness=0.03,
        substrate_index=1.5,
    )
    res = run_grating(
        spec, wl_range=(0.7, 0.95), n_freq=8, angle_deg=0.0, pol="s",
        resolution=resolution, tin_material=meep.Medium(index=2.0),
        decay_by=1e-6,
    )
    worst = float(np.max(np.abs(res.R + res.T - 1.0)))
    bounded = bool(np.all(res.R >= -0.01) and np.all(res.T >= -0.01))
    ok = bounded and worst < 0.03
    print(f"  max |R+T-1| = {worst:.4f}   bounded={bounded}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_defective_grating(resolution, params):
    _section("4. Real TiN finite grating with defects (spectrum)")
    from tin_dispersion import build_meep_medium  # noqa: PLC0415
    import meep  # noqa: PLC0415

    spec = GratingSpec(
        n_ridges=6, period=1.2, amplitude=0.05, base_thickness=0.03,
        substrate_index=1.5,
    ).with_defects(jitter_std=0.15, n_missing=1, rng=np.random.default_rng(0))

    tin = build_meep_medium(params, meep)
    res = run_grating(
        spec, wl_range=(0.6, 1.2), n_freq=10, angle_deg=0.0, pol="s",
        resolution=resolution, tin_material=tin, decay_by=1e-6,
    )
    print("   wl(µm)     R        T        A")
    for wl, R, T in zip(res.wl_um, res.R, res.T):
        print(f"   {wl:6.3f}  {R:7.4f}  {T:7.4f}  {1-R-T:7.4f}")
    finite = bool(np.all(np.isfinite(res.R)) and np.all(np.isfinite(res.T)))
    ok = finite and np.all(res.R >= -0.02) and np.all(res.T >= -0.02)
    print(f"  finite & bounded -> {'PASS' if ok else 'FAIL'}")
    out = os.path.join(_HERE, "..", "reports", "m0_defective_grating.csv")
    res.to_csv(out)
    print(f"  saved: {os.path.normpath(out)}")
    return ok


def main():
    p = argparse.ArgumentParser(description="M0 validation of the FDTD simulator")
    p.add_argument("--dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--resolution", type=float, default=40.0)
    args = p.parse_args()

    wl_range = (0.4, 0.95) if args.dataset == "labo" else (0.5, 1.5)

    results = {}
    ok_tin, params = check_tin(args.dataset, wl_range, args.resolution)
    results["tin"] = ok_tin
    results["fresnel"] = check_fresnel(args.resolution)
    results["energy"] = check_energy(args.resolution)
    results["defective"] = check_defective_grating(args.resolution, params)

    _section("SUMMARY")
    for name, ok in results.items():
        print(f"  {name:12s}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
