"""Re-calibrate archived oblique references after the per-plane flux fix.

Oblique references written before commit ``E5b`` divided T by the incident flux
measured at the *reflection* plane.  At oblique incidence the truncated
plane-wave source also loses energy to the side PML on its way down to the
transmission plane, so the two calibrations differ (3.9 % at 30 deg, up to 90 %
at 60 deg with a narrow cell).  The archived JSON stores everything needed to
recompute the transmission-plane calibration exactly (``fcen``, ``fwidth``,
``sx``, ``sy``, ``y_src``, ``y_refl``, ``y_tran``, ``resolution``, ``dpml``), and
an empty-cell calibration run is cheap compared with a full FDTD reference.

The corrected references are written to a separate directory by default so the
original files remain the record of what was actually measured.

Example
-------
    python scripts/repair_reference_flux.py --angles 30 \
        --data-dir reports/m2/data140 --out-dir reports/m2/data140_fix
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
    p = argparse.ArgumentParser(description="Repair oblique T normalisation")
    p.add_argument("--data-dir", default=os.path.join(
        _HERE, "..", "reports", "m2", "data140"))
    p.add_argument("--out-dir", default=None,
                   help="write corrected copies here (default: <data-dir>_fix)")
    p.add_argument("--angles", type=float, nargs="*", default=[30.0],
                   help="only repair references at these angles")
    p.add_argument("--n-calib", type=int, default=5,
                   help="number of wavelengths at which to run the calibration")
    p.add_argument("--decay-by", type=float, default=1e-7)
    p.add_argument("--d-t", type=float, default=20.0)
    p.add_argument("--energy-tol", type=float, default=0.02)
    p.add_argument("--validate-interp", action="store_true",
                   help="calibrate every wavelength for the first reference and "
                        "report the interpolation error")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N references (debugging)")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to these reference basenames (no extension)")
    return p.parse_args(argv)


def calibration_indices(n: int, k: int):
    """Evenly spaced indices including both ends."""
    if n <= 0:
        return []
    k = max(2, min(int(k), n))
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, k)})


def ratios_for_run(meep, run: dict, pol: str, angle: float, *, decay_by, dT):
    """Recompute ``inc_refl``/``inc_tran`` for one stored per-wavelength run."""
    from meep_grating import empty_cell_fluxes

    data, inc_refl, inc_tran = empty_cell_fluxes(
        meep,
        fcen=float(run["fcen"]),
        fwidth=float(run["fwidth"]),
        n_freq=1,
        angle_deg=float(angle),
        pol=pol,
        sx=float(run["sx"]),
        sy=float(run["sy"]),
        y_src=float(run["y_src"]),
        y_refl=float(run["y_refl"]),
        y_tran=float(run["y_tran"]),
        resolution=float(run["resolution"]),
        dpml=float(run["dpml"]),
        decay_by=decay_by,
        dT=dT,
    )
    ratio = float(np.atleast_1d(inc_refl)[0]) / float(np.atleast_1d(inc_tran)[0])
    archived = float(np.atleast_1d(run.get("incident_flux", np.nan))[0])
    recomputed = float(np.atleast_1d(inc_refl)[0])
    dev = (abs(recomputed - archived) / archived) if archived else float("nan")
    return ratio, dev


def repair_one(path: str, out_dir: str, a, *, full_calibration: bool):
    """Return a summary dict for one reference."""
    base = os.path.basename(path)
    with open(path) as fh:
        meta = json.load(fh)
    runs = meta.get("fdtd_meta", {}).get("runs")
    if not runs:
        return {"file": base, "status": "skip (not oblique)"}

    angle = float(meta["angle_deg"])
    pol = meta["pol"]
    if angle in (0.0,) or angle < 1e-9:
        return {"file": base, "status": "skip (normal incidence)"}
    if a.angles and round(angle) not in {round(x) for x in a.angles}:
        return {"file": base, "status": f"skip (angle {angle})"}

    import meep

    # The oblique generator wrote Lx_um as NaN (the per-run meta is a list).
    if not np.isfinite(float(meta.get("Lx_um", np.nan))):
        meta["Lx_um"] = float(runs[0]["sx"])

    csv_path = os.path.splitext(path)[0] + ".csv"
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    wl, R, T = arr[:, 0], arr[:, 1], arr[:, 2]
    order = np.argsort(wl)
    wl, R, T = wl[order], R[order], T[order]

    fcen = np.array([float(r["fcen"]) for r in runs])[order]
    wl_run = 1.0 / fcen

    idx = (list(range(len(runs))) if full_calibration
           else calibration_indices(len(runs), a.n_calib))
    ratios, devs = [], []
    for i in idx:
        r = ratios_for_run(meep, runs[i], pol, angle,
                           decay_by=a.decay_by, dT=a.d_t)
        ratios.append(r[0])
        devs.append(r[1])
        print(f"    calib wl={wl_run[i]:.3f} ratio={r[0]:.4f} "
              f"(inc_refl archived vs recomputed: {r[1]:.2%})", flush=True)

    if full_calibration:
        ratio_all = np.asarray(ratios)
        interp_err = 0.0
    else:
        ratio_all = np.interp(wl_run, wl_run[idx], ratios)
        interp_err = float("nan")

    T_new = T * ratio_all
    excess = float(np.max(R + T_new - 1.0))
    status = "ok"
    if excess > a.energy_tol:
        status = f"FAIL energy: max(R+T-1)={excess:.3f}"
    if np.nanmax(devs) > 0.01:
        status = f"WARN calibration mismatch {np.nanmax(devs):.2%}"

    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, base.replace(".json", ".csv")),
               np.column_stack([wl, R, T_new]), delimiter=",",
               header="wl_um,R,T", comments="")
    meta["flux_fix"] = {
        "method": "empty-cell inc_tran calibration",
        "n_calib": len(idx),
        "calib_wl_um": [float(wl_run[i]) for i in idx],
        "calib_ratio": [float(x) for x in ratios],
        "ratio": [float(x) for x in ratio_all],
        "archived_vs_recomputed_inc_refl_max_dev": float(np.nanmax(devs)),
        "energy_excess_max": excess,
        "status": status,
    }
    with open(os.path.join(out_dir, base), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"  {base}: ratio {np.min(ratio_all):.3f}-{np.max(ratio_all):.3f}  "
          f"A mean {np.mean(1 - R - T):.3f} -> {np.mean(1 - R - T_new):.3f}  "
          f"[{status}]", flush=True)
    return {"file": base, "status": status, "excess": excess,
            "ratio_min": float(np.min(ratio_all)),
            "ratio_max": float(np.max(ratio_all)),
            "A_mean_before": float(np.mean(1 - R - T)),
            "A_mean_after": float(np.mean(1 - R - T_new)),
            "max_calib_dev": float(np.nanmax(devs)),
            "interp_err": interp_err}


def main(argv=None):
    a = parse(argv)
    data_dir = os.path.normpath(a.data_dir)
    out_dir = os.path.normpath(a.out_dir or (data_dir + "_fix"))
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if a.only:
        wanted = {n if n.endswith(".json") else n + ".json" for n in a.only}
        files = [f for f in files if f in wanted]
    if a.limit:
        files = files[:a.limit]
    print(f"repairing {len(files)} references from {data_dir} -> {out_dir}",
          flush=True)

    summaries = []
    for i, name in enumerate(files):
        full = a.validate_interp and i == 0
        summaries.append(repair_one(os.path.join(data_dir, name), out_dir, a,
                                    full_calibration=full))

    ok = sum(1 for s in summaries if s.get("status") == "ok")
    bad = [s for s in summaries if str(s.get("status", "")).startswith("FAIL")]
    print(f"\n{ok} repaired, {len(bad)} failed, "
          f"{len(summaries) - ok - len(bad)} skipped")
    for s in bad:
        print(f"  FAIL {s['file']}: {s['status']}")
    # With --only this process handles one reference of a parallel batch; a
    # shared summary file would be clobbered by the sibling workers.
    if a.only is None:
        with open(os.path.join(out_dir, "flux_repair_summary.json"), "w") as fh:
            json.dump(summaries, fh, indent=2, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
