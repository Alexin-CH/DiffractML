"""Build the paired MEEP/RCWA dataset for one or more (pol, angle) settings.

For each reference produced by ``generate_m2_references.py`` this runs the fast
RCWA supercell, de-embeds both spectra, and appends feature/target arrays.
Geometry ids are shared across polarisations and angles, so a split by geometry
still means "unseen structure" for every illumination.

Example
-------
    python scripts/build_m2_dataset.py --angles 0 30 60 --pols s p \
        --specs 0 3 5 9 13 15 19 22 --data-dir reports/m2/data140 \
        --out reports/m2/dataset_generalisation.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402

from benchmark import deembed_reference, deembed_rcwa, load_reference  # noqa: E402
from m2_data import (  # noqa: E402
    FEATURE_NAMES, RCWA_CFG, build_example, reference_base,
)
from rcwa_supercell import RCWAConfig, spectrum_supercell  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description="Build the paired dataset")
    p.add_argument("--data-dir", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "data140"))
    p.add_argument("--specs", type=int, nargs="*", default=None,
                   help="spec ids (default: 0..n-specs-1)")
    p.add_argument("--n-specs", type=int, default=24)
    p.add_argument("--angles", type=float, nargs="+", default=[0.0])
    p.add_argument("--pols", nargs="+", default=["s"], choices=["s", "p"])
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2",
                                        "dataset_generalisation.npz"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    cfg = RCWAConfig(device=a.device, **RCWA_CFG)
    ids = a.specs if a.specs else list(range(a.n_specs))

    Xs, Yrs, Yas, Raws = [], [], [], []
    groups, wls, angles, pols = [], [], [], []
    used = []
    for pol in a.pols:
        for angle in a.angles:
            for idx in ids:
                base = os.path.join(a.data_dir,
                                    reference_base(idx, pol, angle))
                if not os.path.exists(base + ".json"):
                    print(f"  [{idx:03d} {pol} {angle:g}deg] missing, skip")
                    continue

                ref = load_reference(base)
                ref_d = deembed_reference(ref)
                spec = ref["spec"]
                wl = ref_d["wl_um"]

                res = spectrum_supercell(
                    spec, wl, cfg, pol=pol, angle_deg=angle,
                    tin_dataset=a.tin_dataset,
                )
                rcwa_d = deembed_rcwa(res, angle, pol, spec.substrate_index)

                X, Yr, Ya, Raw = build_example(spec, wl, rcwa_d, ref_d)
                Xs.append(X)
                Yrs.append(Yr)
                Yas.append(Ya)
                Raws.append(Raw)
                groups.append(np.full(len(wl), idx, dtype=np.int32))
                wls.append(wl.astype(np.float32))
                angles.append(np.full(len(wl), float(angle), dtype=np.float32))
                pols.append(np.full(len(wl), 0 if pol == "s" else 1,
                                    dtype=np.int8))
                used.append((idx, pol, float(angle)))
                print(f"  [{idx:03d} {pol} {angle:g}deg] wl={len(wl)}  "
                      f"raw RMSE(A)="
                      f"{np.sqrt(np.mean((Raw[:, 2] - Ya[:, 2]) ** 2)):.4f}",
                      flush=True)

    if not Xs:
        raise SystemExit("no references found")

    X = np.concatenate(Xs)
    Y_res = np.concatenate(Yrs)
    Y_abs = np.concatenate(Yas)
    RCWA_abs = np.concatenate(Raws)
    group = np.concatenate(groups)
    wl = np.concatenate(wls)
    angle_deg = np.concatenate(angles)
    pol_id = np.concatenate(pols)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out,
        X=X, Y_res=Y_res, Y_abs=Y_abs, RCWA_abs=RCWA_abs,
        group=group, wl=wl, angle_deg=angle_deg, pol_id=pol_id,
        feature_names=np.asarray(FEATURE_NAMES),
        spec_ids=np.asarray(sorted({u[0] for u in used}), dtype=np.int32),
        rcwa_cfg=np.asarray(json.dumps(RCWA_CFG)),
    )
    print(f"\nwrote {os.path.normpath(a.out)}: {X.shape[0]} samples, "
          f"{X.shape[1]} features, {len({u[0] for u in used})} geometries, "
          f"{len(used)} (geometry, pol, angle) sets")


if __name__ == "__main__":
    main()
