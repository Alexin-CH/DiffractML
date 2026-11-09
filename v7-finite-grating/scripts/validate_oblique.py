"""Gate the oblique references: does de-embedded RCWA still track MEEP?

Run after the remote MEEP job finishes.  For every (geometry, angle, pol) it
loads the new reference, de-embeds both sides with the angle/pol Fresnel
baseline, runs the padded supercell (padding 1.0, nh=32), and reports RMSE of
R, T and A.

Pass criterion used in the report: oblique RMSE(A) comparable to normal
incidence (< 0.05 raw), i.e. the aperture scaling did not break.

Example
-------
    python scripts/validate_oblique.py --specs 0 13 --angles 0 30
    python scripts/validate_oblique.py            # all 8 geometries
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

import numpy as np  # noqa: E402

from benchmark import deembed_rcwa, deembed_reference, load_reference  # noqa: E402
from metrics import spectral_rmse  # noqa: E402
from rcwa_supercell import RCWAConfig, spectrum_supercell  # noqa: E402


def base_name(data_dir, idx, pol, angle):
    if abs(angle) < 1e-9:
        return os.path.join(data_dir, f"m2_{idx:03d}_{pol}")
    return os.path.join(data_dir, f"m2_{idx:03d}_{pol}_a{angle:02.0f}")


def parse(argv=None):
    p = argparse.ArgumentParser(description="Validate oblique references")
    p.add_argument("--data-dir", default=os.path.join(
        _HERE, "..", "reports", "m2", "data140"))
    p.add_argument("--specs", type=int, nargs="*", default=None)
    p.add_argument("--angles", type=float, nargs="*", default=[0, 30, 60])
    p.add_argument("--pols", nargs="*", default=["s", "p"])
    p.add_argument("--padding", type=float, default=1.0)
    p.add_argument("--nh", type=int, default=32)
    p.add_argument("--grid-nm", type=float, default=10.0)
    p.add_argument("--n-layers", type=int, default=40)
    p.add_argument("--tin-dataset", choices=["labo", "beliaev"], default="beliaev")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default=os.path.join(
        _HERE, "..", "reports", "m2"))
    p.add_argument("--tag", default="oblique_validation")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    specs = a.specs if a.specs else [0, 3, 5, 9, 13, 15, 19, 22]
    cfg = RCWAConfig(padding_um=a.padding, nh=a.nh, grid_nm=a.grid_nm,
                     n_layers=a.n_layers, device=a.device)

    rows = []
    for idx in specs:
        for pol in a.pols:
            for angle in a.angles:
                base = base_name(a.data_dir, idx, pol, angle)
                if not os.path.exists(base + ".json"):
                    print(f"  [{idx:03d} {pol} {angle:g}] missing, skip")
                    continue
                ref = load_reference(base)
                ref_d = deembed_reference(ref)
                spec = ref["spec"]
                wl = ref_d["wl_um"]

                res = spectrum_supercell(
                    spec, wl, cfg, pol=pol, angle_deg=angle,
                    tin_dataset=a.tin_dataset,
                )
                rc = deembed_rcwa(res, angle, pol, spec.substrate_index)

                row = {
                    "spec": idx, "pol": pol, "angle": float(angle),
                    "R": spectral_rmse(rc["R"], ref_d["R"]),
                    "T": spectral_rmse(rc["T"], ref_d["T"]),
                    "A": spectral_rmse(rc["A"], ref_d["A"]),
                }
                rows.append(row)
                print(f"  [{idx:03d} {pol} {angle:>4g}°] "
                      f"RMSE R={row['R']:.4f} T={row['T']:.4f} "
                      f"A={row['A']:.4f}", flush=True)

    if not rows:
        raise SystemExit("no references found")

    def agg(pred):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return None
        return {k: float(np.mean([r[k] for r in sel]))
                for k in ("R", "T", "A")} | {"n": len(sel)}

    summary = {
        "all": agg(lambda r: True),
        "normal": agg(lambda r: abs(r["angle"]) < 1e-9),
        "oblique": agg(lambda r: abs(r["angle"]) > 1e-9),
    }
    for angle in sorted({r["angle"] for r in rows}):
        summary[f"angle_{angle:g}"] = agg(lambda r, a=angle: r["angle"] == a)
    for pol in sorted({r["pol"] for r in rows}):
        summary[f"pol_{pol}"] = agg(lambda r, p=pol: r["pol"] == p)

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.tag}.json"), "w") as fh:
        json.dump({"config": vars(a), "summary": summary, "rows": rows},
                  fh, indent=2)

    lines = ["# Oblique reference validation", "",
             f"- supercell padding {a.padding} µm, nh={a.nh}, "
             f"grid {a.grid_nm} nm, {a.n_layers} layers",
             f"- references: {os.path.relpath(a.data_dir, os.path.join(_HERE, '..'))}",
             "",
             "## RMSE by subset", "",
             "| subset | n | R | T | A |", "|---|---:|---:|---:|---:|"]
    for key, s in summary.items():
        if s is None:
            continue
        lines.append(f"| {key} | {s['n']} | {s['R']:.4f} | {s['T']:.4f} | "
                     f"{s['A']:.4f} |")
    lines += ["", "## Per case", "",
              "| spec | pol | angle | R | T | A |",
              "|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['spec']:03d} | {r['pol']} | {r['angle']:g} | "
                     f"{r['R']:.4f} | {r['T']:.4f} | {r['A']:.4f} |")
    lines.append("")
    path = os.path.join(a.out_dir, f"{a.tag}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\nRMSE by subset")
    for key, s in summary.items():
        if s is not None:
            print(f"  {key:9s} n={s['n']:2d}  R={s['R']:.4f} "
                  f"T={s['T']:.4f}  A={s['A']:.4f}")
    print(f"wrote {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
