"""Generate M2 MEEP references for a deterministic set of gratings (MEEP env).

For each ``idx`` in ``range(n_specs)`` the spec comes from
``m2_data.make_spec(idx)`` (shared with the dataset builder).  References are
written per spec and the script is resumable.

Example
-------
    python scripts/generate_m2_references.py --n-specs 16 --resolution 105 \
        --wl-range 0.7 1.15 --n-freq 16 --pol s
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from benchmark import save_reference  # noqa: E402
from m2_data import make_spec  # noqa: E402
from meep_grating import run_grating  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description="Generate M2 MEEP references")
    p.add_argument("--n-specs", type=int, default=16)
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "data"))
    p.add_argument("--wl-range", type=float, nargs=2, default=[0.7, 1.15])
    p.add_argument("--n-freq", type=int, default=16)
    p.add_argument("--resolution", type=float, default=105.0)
    p.add_argument("--pol", type=str, default="s", choices=["s", "p"])
    p.add_argument("--pad", type=float, default=1.5)
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--decay-by", type=float, default=1e-7)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main(argv=None):
    a = parse()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"generating {a.n_specs} MEEP references -> {os.path.normpath(a.out_dir)}")

    for idx in range(a.n_specs):
        base = os.path.join(a.out_dir, f"m2_{idx:03d}_{a.pol}")
        if not a.no_resume and os.path.exists(base + ".json"):
            print(f"  [{idx:03d}] exists, skip")
            continue

        spec = make_spec(idx)
        print(f"  [{idx:03d}] ridges={spec.n_ridges} period={spec.period:.3f} "
              f"amp={spec.amplitude:.3f} missing={int(spec.missing.sum())}")
        result = run_grating(
            spec,
            wl_range=tuple(a.wl_range),
            n_freq=a.n_freq,
            pol=a.pol,
            resolution=a.resolution,
            pad=a.pad,
            tin_dataset=a.tin_dataset,
            decay_by=a.decay_by,
        )
        save_reference(result, base, spec=spec,
                       extra={"n_freq": a.n_freq, "wl_range": list(a.wl_range),
                              "stage": "m2"})
        print(f"        saved, A mean={float(np.mean(1 - result.R - result.T)):.4f}")


if __name__ == "__main__":
    main()
