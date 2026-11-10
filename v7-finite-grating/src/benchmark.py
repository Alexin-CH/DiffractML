"""M1 benchmark: padded-supercell RCWA vs the MEEP finite-grating reference.

The benchmark is deliberately split so each half can run in its own environment
(MEEP/torch environments conflict):

* ``scripts/make_reference.py`` (MEEP env) writes a reference CSV + JSON sidecar.
* ``scripts/run_benchmark.py`` (TORCWA/torch env) loads it, runs the RCWA sweep,
  de-embeds both spectra and writes the comparison report.

The primary metric is the **de-embedded absorption** spectrum
``A_grating = (1 - R - T) * Lx / W``, which is baseline-free because the bare
substrate absorbs nothing.  R and T are compared after subtracting the
bare-substrate Fresnel baseline and area-normalising (see :mod:`metrics`).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np

from grating import GratingSpec
from metrics import bare_substrate, compare_spectra, deembed

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REF_DIR = os.path.normpath(os.path.join(_CURRENT_DIR, "..", "reports", "reference"))


# --------------------------------------------------------------------------- #
# GratingSpec (de)serialisation
# --------------------------------------------------------------------------- #
def spec_to_dict(spec: GratingSpec) -> dict:
    d = asdict(spec)
    d["ridge_scale"] = np.asarray(spec.ridge_scale, dtype=float).tolist()
    d["missing"] = np.asarray(spec.missing, dtype=bool).tolist()
    return d


def spec_from_dict(d: dict) -> GratingSpec:
    d = dict(d)
    d["ridge_scale"] = np.asarray(d["ridge_scale"], dtype=float)
    d["missing"] = np.asarray(d["missing"], dtype=bool)
    return GratingSpec(**d)


# --------------------------------------------------------------------------- #
# Reference I/O
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    """Recursively convert numpy/complex values into JSON-safe ones."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_reference(result, path: str, *, spec: GratingSpec, extra: Optional[dict] = None):
    """Save a MEEP ``FDTDResult`` as ``<path>.csv`` + ``<path>.json``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    csv_path = path if path.endswith(".csv") else path + ".csv"
    json_path = os.path.splitext(csv_path)[0] + ".json"

    data = np.column_stack([result.wl_um, result.R, result.T])
    np.savetxt(csv_path, data, delimiter=",", header="wl_um,R,T", comments="")

    meta = {
        "spec": spec_to_dict(spec),
        "Lx_um": float(result.meta.get("sx", np.nan)),
        "W_um": float(spec.aperture_width),
        "angle_deg": float(result.angle_deg),
        "pol": result.pol,
        "n_sub": float(spec.substrate_index),
        "fdtd_meta": result.meta,
    }
    if extra:
        meta.update(extra)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(meta), fh, indent=2, default=str)
    return csv_path, json_path


def load_reference(path: str) -> dict:
    """Load ``<path>.csv`` + ``<path>.json`` into a reference dict."""
    if path.endswith(".json"):
        json_path = path
        csv_path = os.path.splitext(path)[0] + ".csv"
    else:
        csv_path = path if path.endswith(".csv") else path + ".csv"
        json_path = os.path.splitext(csv_path)[0] + ".json"

    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    wl = np.asarray(raw[:, 0], dtype=float)
    order = np.argsort(wl)
    return {
        "wl_um": wl[order],
        "R": np.asarray(raw[order, 1], dtype=float),
        "T": np.asarray(raw[order, 2], dtype=float),
        "A": 1.0 - raw[order, 1] - raw[order, 2],
        "spec": spec_from_dict(meta["spec"]),
        "Lx_um": float(meta["Lx_um"]),
        "W_um": float(meta["W_um"]),
        "angle_deg": float(meta["angle_deg"]),
        "pol": meta["pol"],
        "n_sub": float(meta["n_sub"]),
        "meta": meta,
    }


def deembed_reference(ref: dict) -> dict:
    """Return a copy of the reference with de-embedded R/T/A."""
    R0, T0 = bare_substrate(ref["angle_deg"], ref["pol"], ref["n_sub"])
    R, T, A = deembed(ref["R"], ref["T"], Lx_um=ref["Lx_um"], W_um=ref["W_um"],
                      R0=R0, T0=T0)
    out = dict(ref)
    out.update({"R": R, "T": T, "A": A, "R0": R0, "T0": T0})
    return out


def deembed_rcwa(result: dict, angle_deg: float, pol: str,
                 n_sub: float = 1.5) -> dict:
    R0, T0 = bare_substrate(angle_deg, pol, n_sub)
    R, T, A = deembed(result["R"], result["T"], Lx_um=result["Lx_um"],
                      W_um=result["W_um"], R0=R0, T0=T0)
    out = dict(result)
    out.update({"R_raw": result["R"], "T_raw": result["T"],
                "R": R, "T": T, "A": A, "R0": R0, "T0": T0})
    return out


# --------------------------------------------------------------------------- #
# RCWA sweep (torch env)
# --------------------------------------------------------------------------- #
def run_rcwa_sweep(spec: GratingSpec, wl_grid_um, configs, *, pol: str,
                   angle_deg: float = 0.0, tin_params: Optional[dict] = None,
                   tin_dataset: str = "beliaev", verbose: bool = False):
    """Run each :class:`RCWAConfig` on ``wl_grid_um`` and de-embed the result."""
    from rcwa_supercell import spectrum_supercell

    results = []
    for cfg in configs:
        if verbose:
            print(f"[rcwa] padding={cfg.padding_um:g} µm  nh={cfg.nh}  "
                  f"grid={cfg.grid_nm:g} nm  layers={cfg.n_layers}")
        res = spectrum_supercell(
            spec, wl_grid_um, cfg, angle_deg=angle_deg, pol=pol,
            tin_params=tin_params, tin_dataset=tin_dataset,
        )
        res["config"] = cfg
        results.append(deembed_rcwa(res, angle_deg, pol, spec.substrate_index))
    return results


# --------------------------------------------------------------------------- #
# Comparison / reporting
# --------------------------------------------------------------------------- #
def compare_against_reference(ref_deembedded: dict, rcwa_results, *,
                              prominence=None):
    """Attach comparison metrics to each de-embedded RCWA result."""
    rows = []
    for res in rcwa_results:
        metrics = compare_spectra(ref_deembedded, res, prominence=prominence)
        cfg = res["config"]
        row = {
            "padding_um": cfg.padding_um,
            "nh": cfg.nh,
            "grid_nm": cfg.grid_nm,
            "n_layers": cfg.n_layers,
            "Lx_um": res["Lx_um"],
            "rmse_A": metrics["A"]["rmse"],
            "rmse_R": metrics["R"]["rmse"],
            "rmse_T": metrics["T"]["rmse"],
            "max_abs_A": metrics["A"]["max_abs"],
            "n_res_ref": metrics["resonance"]["n_ref"],
            "n_res_test": metrics["resonance"]["n_test"],
            "n_res_matched": metrics["resonance"]["n_matched"],
            "mean_dwl_um": metrics["resonance"]["mean_dwl"],
            "mean_dwidth_um": metrics["resonance"]["mean_dwidth"],
        }
        rows.append(row)
    return rows


def write_report(rows, ref_deembedded: dict, *, csv_path: str, md_path: str,
                 plot_path: Optional[str] = None):
    """Write the comparison table (CSV + Markdown) and an optional plot."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    header = ",".join(fieldnames)
    data = "\n".join(",".join(str(r[k]) for k in fieldnames) for r in rows)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n" + data + "\n")

    best = min(rows, key=lambda r: r["rmse_A"]) if rows else None
    spec = ref_deembedded["spec"]
    lines = [
        "# M1 benchmark — padded-supercell RCWA vs MEEP finite grating",
        "",
        f"- ridges: {spec.n_ridges}, period: {spec.period:g} µm, "
        f"amplitude: {spec.amplitude:g} µm, base: {spec.base_thickness:g} µm",
        f"- defects: jitter_std≈{float(np.std(spec.ridge_scale)):.3f}, "
        f"missing={int(np.sum(spec.missing))}",
        f"- polarisation: {ref_deembedded['pol']}, "
        f"angle: {ref_deembedded['angle_deg']:g}°, "
        f"substrate n={ref_deembedded['n_sub']:g}",
        f"- MEEP reference: Lx={ref_deembedded['Lx_um']:.2f} µm "
        f"(W={ref_deembedded['W_um']:.2f} µm), "
        f"resolution={ref_deembedded['meta'].get('fdtd_meta', {}).get('resolution', '?')} px/µm",
        "",
        "Primary metric: RMSE of the de-embedded **absorption** spectrum "
        "`A=(1-R-T)·Lx/W` (baseline-free). R/T RMSE are after subtracting the "
        "bare-substrate Fresnel baseline.",
        "",
    ]
    if best is not None:
        lines += [
            f"**Best config:** padding={best['padding_um']:g} µm, nh={best['nh']}, "
            f"RMSE(A)={best['rmse_A']:.4f}, RMSE(R)={best['rmse_R']:.4f}, "
            f"RMSE(T)={best['rmse_T']:.4f}.",
            "",
        ]
    lines += [
        "| padding (µm) | nh | grid (nm) | RMSE A | RMSE R | RMSE T | max|ΔA| | matched res | mean Δλ (µm) | mean Δwidth (µm) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['padding_um']:g} | {r['nh']} | {r['grid_nm']:g} | "
            f"{r['rmse_A']:.4f} | {r['rmse_R']:.4f} | {r['rmse_T']:.4f} | "
            f"{r['max_abs_A']:.4f} | {r['n_res_matched']} | "
            f"{r['mean_dwl_um']:.4f} | {r['mean_dwidth_um']:.4f} |"
        )
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    if plot_path is not None and rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        # Rebuild the best config's spectrum for plotting (caller passes results
        # via ref_deembedded['_best_rcwa'] when available).
        best_res = ref_deembedded.get("_best_rcwa")
        if best_res is None:
            return
        wl = ref_deembedded["wl_um"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(wl, ref_deembedded["A"], "k-", label="MEEP")
        ax[0].plot(best_res["wl_um"], best_res["A"], "r--", label="RCWA (best)")
        ax[0].set_xlabel("wavelength (µm)")
        ax[0].set_ylabel("absorption A")
        ax[0].legend()
        ax[1].plot(wl, ref_deembedded["R"], "k-", label="MEEP R")
        ax[1].plot(best_res["wl_um"], best_res["R"], "r--", label="RCWA R")
        ax[1].plot(wl, ref_deembedded["T"], "b-", label="MEEP T")
        ax[1].plot(best_res["wl_um"], best_res["T"], "c--", label="RCWA T")
        ax[1].set_xlabel("wavelength (µm)")
        ax[1].legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Shared CLI helpers
# --------------------------------------------------------------------------- #
def add_spec_args(parser):
    """Add the finite-grating geometry options shared by both scripts."""
    parser.add_argument("--n-ridges", type=int, default=8)
    parser.add_argument("--period", type=float, default=1.2, help="µm")
    parser.add_argument("--amplitude", type=float, default=0.05, help="µm")
    parser.add_argument("--base-thickness", type=float, default=0.03, help="µm")
    parser.add_argument("--substrate-index", type=float, default=1.5)
    parser.add_argument("--jitter-std", type=float, default=0.0,
                        help="per-ridge height jitter std (0 = perfect)")
    parser.add_argument("--n-missing", type=int, default=0,
                        help="number of missing ridges")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def spec_from_args(a) -> GratingSpec:
    spec = GratingSpec(
        n_ridges=a.n_ridges,
        period=a.period,
        amplitude=a.amplitude,
        base_thickness=a.base_thickness,
        substrate_index=a.substrate_index,
    )
    if a.jitter_std > 0 or a.n_missing > 0:
        spec = spec.with_defects(
            jitter_std=a.jitter_std, n_missing=a.n_missing,
            rng=np.random.default_rng(a.seed),
        )
    return spec
