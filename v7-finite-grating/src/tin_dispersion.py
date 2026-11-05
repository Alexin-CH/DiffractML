"""Fit tabulated TiN n/k data to a MEEP-compatible Drude-Lorentz model.

MEEP can only ingest frequency-dependent permittivity as a sum of Lorentz /
Drude oscillators (see ``meep.Medium``).  This module:

1. loads the tabulated ``n, k`` data shipped with the TORCWA submodule,
2. fits ``eps(f) = eps_inf + Drude + sum(Lorentz)`` to ``(n + ik)^2``,
3. verifies the fit against the tabulated ``n, k``,
4. builds a ``meep.Medium`` whose ``epsilon(f)`` reproduces the fitted model.

Units
-----
Frequencies are in 1/µm (MEEP's default ``a = 1 µm``), so ``f = 1 / λ_um``.
Sign convention is MEEP's ``exp(-iωt)`` with positive imaginary part = loss.

Model form (Meep materials documentation)
----------------------------------------
    eps(f) = eps_inf
             + sigma_p * f_p^2 / (f_p^2 - f^2 - i f gamma_p)          (Drude)
             + sum_j sigma_j f_j^2 / (f_j^2 - f^2 - i f gamma_j)      (Lorentz)
"""

from __future__ import annotations

import json
import os
from typing import Mapping, Optional

import numpy as np

_CSV_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "TORCWA", "simulations", "sin_tin")
)
_DATASETS = {
    "labo": "TiN-RefIdx-labo.csv",
    "beliaev": "TiN-RefIdx-Beliaev-sputtering.csv",
}


# --------------------------------------------------------------------------- #
# Tabulated data
# --------------------------------------------------------------------------- #
def load_nk(dataset: str = "labo") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load tabulated TiN ``(wl_um, n, k)``, sorted by wavelength.

    ``labo`` is measured in nm, ``beliaev`` in µm; both are returned in µm.
    """
    if dataset not in _DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(_DATASETS)}")
    path = os.path.join(_CSV_DIR, _DATASETS[dataset])

    with open(path, encoding="utf-8") as fh:
        header = fh.readline().lower()
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    wl = np.asarray(raw[:, 0], dtype=float)
    n = np.asarray(raw[:, 1], dtype=float)
    k = np.asarray(raw[:, 2], dtype=float)

    # The labo header is "wl (nm)" (µm otherwise).  Detect from the header.
    if "nm" in header and "microm" not in header:
        wl = wl / 1000.0

    order = np.argsort(wl)
    return wl[order], n[order], k[order]


def nk_to_epsilon(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Complex permittivity from ``n + ik``: ``eps = (n + i k)^2``."""
    return (np.asarray(n) + 1j * np.asarray(k)) ** 2


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _lorentz(sigma: float, f0: float, gamma: float, f: np.ndarray) -> np.ndarray:
    """MEEP ``LorentzianSusceptibility``: ``sigma f0^2 / (f0^2 - f^2 - i f gamma)``."""
    return sigma * f0**2 / (f0**2 - f**2 - 1j * f * gamma)


def _drude(sigma: float, f0: float, gamma: float, f: np.ndarray) -> np.ndarray:
    """MEEP ``DrudeSusceptibility``: ``i sigma f0^2 / (f (gamma - i f))``."""
    return 1j * sigma * f0**2 / (f * (gamma - 1j * f))


def epsilon_model(params: Mapping, f: np.ndarray) -> np.ndarray:
    """Evaluate the Drude-Lorentz permittivity model at frequencies ``f``.

    The oscillator forms match MEEP's ``LorentzianSusceptibility`` and
    ``DrudeSusceptibility`` exactly, so ``Medium.epsilon(f)`` reproduces this
    function (verified by ``verify_medium``).
    """
    f = np.asarray(f, dtype=float)
    eps = np.full(f.shape, float(params["eps_inf"]), dtype=complex)

    drude = params.get("drude")
    if drude is not None:
        eps = eps + _drude(drude["sigma"], drude["f"], drude["gamma"], f)

    for osc in params.get("lorentz", []):
        eps = eps + _lorentz(osc["sigma"], osc["f"], osc["gamma"], f)
    return eps


def _unpack(theta: np.ndarray, n_osc: int, with_drude: bool) -> dict:
    """Turn a flat parameter vector into the dict consumed by ``epsilon_model``."""
    i = 0
    eps_inf = theta[i]
    i += 1
    params = {"eps_inf": float(eps_inf), "lorentz": []}
    if with_drude:
        params["drude"] = {
            "sigma": float(theta[i]),
            "f": float(theta[i + 1]),
            "gamma": float(theta[i + 2]),
        }
        i += 3
    for _ in range(n_osc):
        params["lorentz"].append(
            {
                "sigma": float(theta[i]),
                "f": float(theta[i + 1]),
                "gamma": float(theta[i + 2]),
            }
        )
        i += 3
    return params


def _pack(params: Mapping, n_osc: int, with_drude: bool) -> np.ndarray:
    theta = [float(params["eps_inf"])]
    if with_drude:
        d = params["drude"]
        theta += [d["sigma"], d["f"], d["gamma"]]
    for osc in params["lorentz"]:
        theta += [osc["sigma"], osc["f"], osc["gamma"]]
    assert len(theta) == 1 + 3 * (n_osc + int(with_drude))
    return np.asarray(theta, dtype=float)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_drude_lorentz(
    wl_um: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    *,
    n_osc: int = 4,
    with_drude: bool = True,
    wl_range: Optional[tuple[float, float]] = None,
    n_restarts: int = 4,
    seed: int = 0,
) -> dict:
    """Least-squares fit of the Drude-Lorentz model to complex permittivity.

    Returns a dict with the fitted parameters, the frequency range, the
    normalised RMSE in ``n, k``, and the fit configuration.
    """
    from scipy.optimize import least_squares

    wl_um = np.asarray(wl_um, dtype=float)
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)

    if wl_range is not None:
        lo, hi = sorted(wl_range)
        mask = (wl_um >= lo) & (wl_um <= hi)
        if mask.sum() < 10:
            raise ValueError("fewer than 10 data points inside wl_range")
        wl_um, n, k = wl_um[mask], n[mask], k[mask]

    # Fit on a log-spaced subsample so each decade contributes equally.
    f_all = 1.0 / wl_um
    f_min, f_max = float(f_all.min()), float(f_all.max())
    target = nk_to_epsilon(n, k)

    n_pts = min(400, f_all.size)
    f_fit = np.geomspace(f_min, f_max, n_pts)
    # Interpolate the tabulated permittivity onto the log grid.
    eps_re = np.interp(f_fit, f_all[::-1], target.real[::-1])
    eps_im = np.interp(f_fit, f_all[::-1], target.imag[::-1])
    eps_fit = eps_re + 1j * eps_im

    eps_scale = max(1.0, float(np.max(np.abs(eps_fit))))

    def residual(theta):
        eps = epsilon_model(_unpack(theta, n_osc, with_drude), f_fit)
        r = np.concatenate([(eps.real - eps_fit.real), (eps.imag - eps_fit.imag)])
        return r / eps_scale

    # Bounds: eps_inf positive; oscillator strengths +-50; resonances inside a
    # generous band; damping positive.
    lo = [0.5]
    hi = [20.0]
    if with_drude:
        lo += [-200.0, 0.05 * f_min, 1e-4]
        hi += [200.0, 4.0 * f_max, 10.0]
    for _ in range(n_osc):
        lo += [-50.0, 0.05 * f_min, 1e-4]
        hi += [50.0, 4.0 * f_max, 10.0]
    lo = np.asarray(lo)
    hi = np.asarray(hi)

    rng = np.random.default_rng(seed)
    best = None
    f_grid = np.geomspace(max(0.1 * f_min, 1e-3), 0.9 * f_max, n_osc)
    for _ in range(max(1, n_restarts)):
        theta0 = [1.5]
        if with_drude:
            theta0 += [5.0, 0.1 * f_min, 0.05 * f_min]
        for f0 in f_grid:
            theta0 += [
                float(rng.uniform(0.2, 3.0)),
                float(f0 * rng.uniform(0.7, 1.3)),
                float(rng.uniform(0.02, 0.3) * (f_max - f_min)),
            ]
        theta0 = np.clip(np.asarray(theta0), lo + 1e-6, hi - 1e-6)

        res = least_squares(
            residual,
            theta0,
            bounds=(lo, hi),
            method="trf",
            max_nfev=20000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        if best is None or res.cost < best.cost:
            best = res

    params = _unpack(best.x, n_osc, with_drude)

    # Verify on the *full* tabulated grid (not just the fit subsample).
    f_full = 1.0 / wl_um
    eps_hat = epsilon_model(params, f_full)
    n_hat = np.sqrt(np.abs(eps_hat)) * np.sqrt(
        0.5 * (1.0 + np.real(eps_hat) / np.abs(eps_hat))
    )
    k_hat = np.sqrt(np.abs(eps_hat)) * np.sqrt(
        0.5 * (1.0 - np.real(eps_hat) / np.abs(eps_hat))
    )
    # Choose the physical branch (n, k >= 0).
    n_hat = np.abs(n_hat)
    k_hat = np.abs(k_hat)
    n_rmse = float(np.sqrt(np.mean((n_hat - n) ** 2)))
    k_rmse = float(np.sqrt(np.mean((k_hat - k) ** 2)))
    n_rel = n_rmse / max(1e-8, float(np.sqrt(np.mean(n**2))))
    k_rel = k_rmse / max(1e-8, float(np.sqrt(np.mean(k**2))))

    params["fit"] = {
        "n_osc": n_osc,
        "with_drude": bool(with_drude),
        "wl_range_um": [float(wl_um.min()), float(wl_um.max())],
        "freq_range": [f_min, f_max],
        "n_rmse": n_rmse,
        "k_rmse": k_rmse,
        "n_rel_rmse": n_rel,
        "k_rel_rmse": k_rel,
        "cost": float(best.cost),
    }
    return params


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_fit(params: Mapping, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)


def load_fit(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_or_fit(
    dataset: str = "labo",
    *,
    cache_path: Optional[str] = None,
    **fit_kwargs,
) -> dict:
    """Return a cached fit if present, otherwise fit and optionally cache it."""
    if cache_path and os.path.exists(cache_path):
        return load_fit(cache_path)
    wl, n, k = load_nk(dataset)
    params = fit_drude_lorentz(wl, n, k, **fit_kwargs)
    params["dataset"] = dataset
    if cache_path:
        save_fit(params, cache_path)
    return params


# --------------------------------------------------------------------------- #
# MEEP medium
# --------------------------------------------------------------------------- #
def build_meep_medium(params: Mapping, meep_module=None):
    """Build a ``meep.Medium`` from fitted Drude-Lorentz parameters.

    ``meep_module`` defaults to ``import meep``; it is injectable so the rest of
    this module stays importable without MEEP (e.g. for fitting on a laptop).
    """
    if meep_module is None:
        import meep as meep_module  # noqa: PLC0415

    susceptibilities = []
    drude = params.get("drude")
    if drude is not None:
        susceptibilities.append(
            meep_module.DrudeSusceptibility(
                frequency=drude["f"],
                gamma=drude["gamma"],
                sigma=drude["sigma"],
            )
        )
    for osc in params.get("lorentz", []):
        susceptibilities.append(
            meep_module.LorentzianSusceptibility(
                frequency=osc["f"],
                gamma=osc["gamma"],
                sigma=osc["sigma"],
            )
        )

    medium = meep_module.Medium(
        epsilon=float(params["eps_inf"]),
        E_susceptibilities=susceptibilities,
    )
    return medium


def verify_medium(params: Mapping, meep_module=None, n_check: int = 25) -> dict:
    """Compare ``meep.Medium.epsilon(f)`` against ``epsilon_model``.

    Returns the maximum relative discrepancy over the fitted band.  A large
    value signals an oscillator-sign/convention mismatch between this module
    and the installed MEEP version.
    """
    if meep_module is None:
        import meep as meep_module  # noqa: PLC0415

    medium = build_meep_medium(params, meep_module)
    f_lo, f_hi = params["fit"]["freq_range"]
    freqs = np.geomspace(f_lo, f_hi, n_check)

    max_rel = 0.0
    rows = []
    for f in freqs:
        eps_meep = np.asarray(medium.epsilon(float(f)))[0, 0]
        eps_ref = epsilon_model(params, np.array([float(f)]))[0]
        denom = max(1e-8, abs(eps_ref))
        rel = abs(eps_meep - eps_ref) / denom
        max_rel = max(max_rel, float(rel))
        rows.append((float(f), complex(eps_meep), complex(eps_ref), float(rel)))
    return {"max_rel": max_rel, "rows": rows}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Fit TiN dispersion to Drude-Lorentz")
    parser.add_argument("--dataset", choices=sorted(_DATASETS), default="labo")
    parser.add_argument("--n-osc", type=int, default=4)
    parser.add_argument("--no-drude", action="store_true")
    parser.add_argument("--wl-range", type=float, nargs=2, default=None,
                        help="fit band in µm (default: full tabulated range)")
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    wl, n, k = load_nk(args.dataset)
    print(f"[{args.dataset}] {wl.size} points, "
          f"wl {wl.min():.3f}-{wl.max():.3f} µm")
    params = fit_drude_lorentz(
        wl, n, k,
        n_osc=args.n_osc,
        with_drude=not args.no_drude,
        wl_range=args.wl_range,
        n_restarts=args.restarts,
        seed=args.seed,
    )
    params["dataset"] = args.dataset
    fit = params["fit"]
    print(f"fit: n_rel_rmse={fit['n_rel_rmse']:.4f}  "
          f"k_rel_rmse={fit['k_rel_rmse']:.4f}  "
          f"band {fit['wl_range_um'][0]:.2f}-{fit['wl_range_um'][1]:.2f} µm")

    out = args.output or os.path.join(
        os.path.dirname(__file__), "..", "data", f"tin_fit_{args.dataset}.json"
    )
    save_fit(params, out)
    print(f"saved: {os.path.normpath(out)}")

    try:
        report = verify_medium(params)
        print(f"meep.Medium vs model max rel eps error: {report['max_rel']:.3e}")
    except ImportError:
        print("meep not installed; skipped Medium verification")


if __name__ == "__main__":
    _main()
