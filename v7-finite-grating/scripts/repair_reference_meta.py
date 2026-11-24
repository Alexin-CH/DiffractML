"""Repair reference metadata produced by the oblique MEEP generator.

Oblique references store their per-wavelength metadata as ``fdtd_meta.runs``
(a list) instead of a flat dict, so the top-level ``Lx_um`` was written as NaN
and the de-embedding silently produced NaN spectra.  This fills ``Lx_um`` from
``fdtd_meta.runs[0].sx`` (all runs share the same cell) and is idempotent.

Example
-------
    python scripts/repair_reference_meta.py --data-dir reports/m2/data140
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os


def parse(argv=None):
    p = argparse.ArgumentParser(description="Repair reference Lx_um")
    p.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reports", "m2",
        "data140"))
    return p.parse_args(argv)


def _sx(meta):
    if "sx" in meta:
        return float(meta["sx"])
    runs = meta.get("runs")
    if runs:
        return float(runs[0].get("sx", math.nan))
    return math.nan


def main(argv=None):
    a = parse(argv)
    fixed, ok, bad = 0, 0, 0
    for path in sorted(glob.glob(os.path.join(a.data_dir, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        lx = d.get("Lx_um")
        if lx is not None and math.isfinite(float(lx)):
            ok += 1
            continue
        sx = _sx(d.get("fdtd_meta", {}))
        if not math.isfinite(sx):
            bad += 1
            print(f"  [skip] {os.path.basename(path)}: no sx in meta")
            continue
        d["Lx_um"] = sx
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)
        fixed += 1

    print(f"{fixed} repaired, {ok} already valid, {bad} unrecoverable "
          f"in {a.data_dir}")


if __name__ == "__main__":
    main()
