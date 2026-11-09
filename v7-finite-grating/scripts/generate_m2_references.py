"""Generate MEEP references for arbitrary (spec set, pol, angle) jobs.

Extends the M2 generator with ``--angle``, a ``--specs`` subset and a
``--spec-set`` choice, keeping the original filenames for the existing
s-pol, normal-incidence data so old references are still picked up.

Naming
------
- M2, angle 0:      ``m2_<idx>_<pol>``            (unchanged)
- M2, oblique:      ``m2_<idx>_<pol>_a<angle>``
- resonant set:     ``res_<idx>_<pol>_a<angle>``  (angle always included)

Example
-------
    # p-pol, normal incidence, all 24 geometries, at reference resolution
    python scripts/generate_m2_references.py --spec-set m2 --pol p --angle 0 \
        --resolution 140 --out-dir reports/m2/data140

    # oblique, subset, both pols
    python scripts/generate_m2_references.py --spec-set m2 --specs 0 3 5 9 13 15 19 22 \
        --pol s --angle 30 --resolution 140 --out-dir reports/m2/data140
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from benchmark import save_reference  # noqa: E402
from m2_data import make_resonant_spec, make_spec  # noqa: E402
from meep_grating import run_grating  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description="Generate MEEP references")
    p.add_argument("--spec-set", choices=["m2", "resonant"], default="m2")
    p.add_argument("--n-specs", type=int, default=16)
    p.add_argument("--specs", type=int, nargs="*", default=None,
                   help="explicit spec ids (overrides --n-specs)")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "data"))
    p.add_argument("--wl-range", type=float, nargs=2, default=[0.7, 1.15])
    p.add_argument("--n-freq", type=int, default=16)
    p.add_argument("--resolution", type=float, default=105.0)
    p.add_argument("--pol", type=str, default="s", choices=["s", "p"])
    p.add_argument("--angle", type=float, default=0.0, help="incidence angle (deg)")
    p.add_argument("--pad", type=float, default=1.5)
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--decay-by", type=float, default=1e-7)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the jobs without running MEEP")
    return p.parse_args(argv)


def reference_base(a, idx: int) -> str:
    if a.spec_set == "m2" and abs(a.angle) < 1e-9:
        return f"m2_{idx:03d}_{a.pol}"
    tag = "m2" if a.spec_set == "m2" else "res"
    return f"{tag}_{idx:03d}_{a.pol}_a{a.angle:02.0f}"


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    ids = a.specs if a.specs else list(range(a.n_specs))
    maker = make_spec if a.spec_set == "m2" else make_resonant_spec
    print(f"{a.spec_set}: {len(ids)} specs, pol={a.pol}, angle={a.angle}°, "
          f"res={a.resolution} -> {os.path.normpath(a.out_dir)}")

    for idx in ids:
        base = os.path.join(a.out_dir, reference_base(a, idx))
        if a.dry_run:
            print(f"  [dry] {base}")
            continue
        if not a.no_resume and os.path.exists(base + ".json"):
            print(f"  [{idx:03d}] exists, skip")
            continue

        spec = maker(idx)
        print(f"  [{idx:03d}] ridges={spec.n_ridges} period={spec.period:.3f} "
              f"amp={spec.amplitude:.3f} missing={int(spec.missing.sum())}")
        t0 = time.time()
        result = run_grating(
            spec,
            wl_range=tuple(a.wl_range),
            n_freq=a.n_freq,
            pol=a.pol,
            resolution=a.resolution,
            angle_deg=a.angle,
            pad=a.pad,
            tin_dataset=a.tin_dataset,
            decay_by=a.decay_by,
        )
        elapsed = time.time() - t0
        save_reference(result, base, spec=spec,
                       extra={"n_freq": a.n_freq, "wl_range": list(a.wl_range),
                              "stage": a.spec_set, "angle_deg": a.angle,
                              "elapsed_s": round(elapsed, 1),
                              "resolution": a.resolution})
        print(f"        saved in {elapsed:.0f}s, "
              f"A mean={float(np.mean(1 - result.R - result.T)):.4f}")


if __name__ == "__main__":
    main()
