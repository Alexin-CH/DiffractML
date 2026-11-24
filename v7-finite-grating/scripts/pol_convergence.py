"""Is the p-pol discrepancy at normal incidence an FDTD or an RCWA artefact?

At 0 deg p-pol the padded-supercell RCWA converges (nh 32 -> 160) to an
absorption ~0.055 above the MEEP reference, while MEEP itself moves *down* as
its resolution increases.  Neither side is obviously converged, so measure both
sides of the question:

* resolution sweep (140 / 200 / 280 px/um) of the FDTD reference: if MEEP is
  converging to the RCWA value, the reference was under-resolved; if it drifts
  further away, the RCWA TM limit is wrong.
* contrast ablation: replace the lossy high-index TiN by a lossless dielectric
  (eps = 4, 9).  If p-pol agrees at low contrast and diverges only with TiN,
  the problem is the treatment of the strongly lossy interface, not the TM
  formulation in general.

This part only needs the MEEP environment; the matching RCWA table is produced
by ``scripts/pol_convergence_rcwa.py`` in the torch environment.

Example
-------
    python scripts/pol_convergence.py --spec 3 --resolutions 140 200 280
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description="p-pol MEEP convergence + ablation")
    p.add_argument("--spec", type=int, default=3)
    p.add_argument("--spec-set", choices=["m2", "resonant"], default="m2")
    p.add_argument("--angle", type=float, default=0.0)
    p.add_argument("--pol", default="p", choices=["s", "p"])
    p.add_argument("--resolutions", type=float, nargs="+", default=[140.0, 200.0, 280.0])
    p.add_argument("--ablation-eps", type=float, nargs="+", default=[4.0, 9.0])
    p.add_argument("--ablation-pols", nargs="+", default=["s", "p"])
    p.add_argument("--ablation-res", type=float, default=140.0)
    p.add_argument("--wl-range", type=float, nargs=2, default=[0.7, 1.15])
    p.add_argument("--n-freq", type=int, default=16)
    p.add_argument("--out-dir", default=os.path.join(_HERE, "..", "reports", "m2",
                                                     "pol_study"))
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    import meep

    from meep_grating import run_grating
    if a.spec_set == "m2":
        from m2_data import make_spec as maker
    else:
        from m2_data import make_resonant_spec as maker

    spec = maker(a.spec)
    cases = []
    for res in a.resolutions:
        cases.append({"tag": f"tin_{a.pol}_res{res:g}", "res": res,
                      "pol": a.pol, "eps": None})
    for eps in a.ablation_eps:
        for pol in a.ablation_pols:
            cases.append({"tag": f"eps{eps:g}_{pol}_res{a.ablation_res:g}",
                          "res": a.ablation_res, "pol": pol, "eps": eps})

    summary = []
    for c in cases:
        path = os.path.join(a.out_dir, c["tag"] + ".npz")
        if a.skip_existing and os.path.exists(path):
            print(f"  {c['tag']}: exists, skip", flush=True)
            summary.append({"tag": c["tag"], **{k: c[k] for k in ("res", "pol", "eps")}})
            continue
        t0 = time.time()
        material = (None if c["eps"] is None
                    else meep.Medium(index=math.sqrt(c["eps"])))
        r = run_grating(
            spec, wl_range=tuple(a.wl_range), n_freq=a.n_freq, pol=c["pol"],
            angle_deg=a.angle, resolution=c["res"], tin_material=material,
        )
        np.savez(path, wl=r.wl_um, R=r.R, T=r.T,
                 spec_id=a.spec, pol=c["pol"], angle=a.angle,
                 resolution=c["res"], eps=-1.0 if c["eps"] is None else c["eps"])
        dt = time.time() - t0
        print(f"  {c['tag']}: A mean={float(np.mean(1 - r.R - r.T)):.4f} "
              f"({dt / 60:.1f} min)", flush=True)
        summary.append({"tag": c["tag"], "res": c["res"], "pol": c["pol"],
                        "eps": c["eps"], "A_mean": float(np.mean(1 - r.R - r.T)),
                        "elapsed_s": dt})

    with open(os.path.join(a.out_dir, "pol_study_meep.json"), "w") as fh:
        json.dump({"spec": a.spec, "angle": a.angle, "cases": summary}, fh,
                  indent=2)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
