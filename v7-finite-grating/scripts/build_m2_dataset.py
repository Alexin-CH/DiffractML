"""Build the M2 paired dataset (torch env).

For each MEEP reference produced by ``generate_m2_references.py`` this runs the
fast RCWA supercell, de-embeds both spectra, and assembles feature/target
arrays into ``dataset.npz``.

Example
-------
    python scripts/build_m2_dataset.py --n-specs 16 --pol s
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from benchmark import deembed_reference, load_reference  # noqa: E402
from m2_data import FEATURE_NAMES, RCWA_CFG, build_example  # noqa: E402
from rcwa_supercell import RCWAConfig, spectrum_supercell  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description="Build M2 paired dataset")
    p.add_argument("--n-specs", type=int, default=16)
    p.add_argument("--data-dir", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "data"))
    p.add_argument("--pol", type=str, default="s", choices=["s", "p"])
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str,
                   default=os.path.join(_HERE, "..", "reports", "m2", "dataset.npz"))
    return p.parse_args()


def main(argv=None):
    a = parse()
    cfg = RCWAConfig(device=a.device, **RCWA_CFG)

    Xs, Yrs, Yas, Raws, groups, wls = [], [], [], [], [], []
    used = []
    for idx in range(a.n_specs):
        base = os.path.join(a.data_dir, f"m2_{idx:03d}_{a.pol}")
        if not os.path.exists(base + ".json"):
            print(f"  [{idx:03d}] missing reference, skip")
            continue

        ref = load_reference(base)
        ref_d = deembed_reference(ref)
        spec = ref["spec"]
        wl = ref_d["wl_um"]

        res = spectrum_supercell(
            spec, wl, cfg, pol=a.pol, angle_deg=ref["angle_deg"],
            tin_dataset=a.tin_dataset,
        )
        from benchmark import deembed_rcwa

        rcwa_d = deembed_rcwa(res, ref["angle_deg"], a.pol, spec.substrate_index)

        X, Yr, Ya, Raw = build_example(spec, wl, rcwa_d, ref_d)
        Xs.append(X)
        Yrs.append(Yr)
        Yas.append(Ya)
        Raws.append(Raw)
        groups.append(np.full(len(wl), idx, dtype=np.int32))
        wls.append(wl.astype(np.float32))
        used.append(idx)
        print(f"  [{idx:03d}] wl={len(wl)}  raw RMSE(A)="
              f"{np.sqrt(np.mean((Raw[:, 2] - Ya[:, 2]) ** 2)):.4f}")

    if not Xs:
        raise SystemExit("no references found")

    X = np.concatenate(Xs)
    Y_res = np.concatenate(Yrs)
    Y_abs = np.concatenate(Yas)
    RCWA_abs = np.concatenate(Raws)
    group = np.concatenate(groups)
    wl = np.concatenate(wls)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out,
        X=X, Y_res=Y_res, Y_abs=Y_abs, RCWA_abs=RCWA_abs,
        group=group, wl=wl,
        feature_names=np.asarray(FEATURE_NAMES),
        spec_ids=np.asarray(used, dtype=np.int32),
        rcwa_cfg=np.asarray(json.dumps(RCWA_CFG)),
    )
    print(f"\nwrote {os.path.normpath(a.out)}: "
          f"{X.shape[0]} samples, {X.shape[1]} features, {len(used)} specs")


if __name__ == "__main__":
    main()
