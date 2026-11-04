"""Spectral comparison metrics for the finite-grating benchmark (pure NumPy).

De-embedding
------------
The FDTD/MEEP and RCWA cells both contain the aperture plus air-over-substrate
padding.  The bare substrate is lossless, so it contributes ``R0 + T0 = 1`` and
*zero* absorption.  Therefore the area-normalised absorption is an exact,
baseline-free grating signal::

    A_grating = (1 - R_total - T_total) * (Lx / W)

For R/T we subtract the bare-substrate Fresnel values and area-normalise::

    R_grating = R0 + (R_total - R0) * (Lx / W)
    T_grating = T0 + (T_total - T0) * (Lx / W)

This recovers the aperture's absolute response (``R + T + A = 1``) and removes
the dilution from the padding.
"""

from __future__ import annotations

import numpy as np

from fresnel import fresnel_RT


def bare_substrate(angle_deg: float, pol: str, n_sub: float = 1.5):
    """Bare-substrate ``(R0, T0)`` for the given incidence and polarisation."""
    R0, T0 = fresnel_RT(1.0, n_sub, angle_deg, pol)
    return float(R0), float(T0)


def deembed(R_total, T_total, *, Lx_um: float, W_um: float,
            R0: float, T0: float):
    """Area-normalise total-cell R/T/A to the aperture (see module docstring)."""
    f = float(Lx_um) / float(W_um)
    R = np.asarray(R_total, dtype=float)
    T = np.asarray(T_total, dtype=float)
    R_g = R0 + (R - R0) * f
    T_g = T0 + (T - T0) * f
    A_g = (1.0 - R - T) * f
    return R_g, T_g, A_g


def spectral_rmse(a, b) -> float:
    """Root-mean-square error between two equally sampled spectra."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def spectral_max_abs(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)))


def interpolate_to(wl_src, y_src, wl_dst):
    """Interpolate a spectrum onto a (possibly unordered) wavelength grid."""
    wl_src = np.asarray(wl_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    order = np.argsort(wl_src)
    return np.interp(np.asarray(wl_dst, dtype=float), wl_src[order], y_src[order])


# --------------------------------------------------------------------------- #
# Resonance positions and widths
# --------------------------------------------------------------------------- #
def find_peaks(wl, y, *, prominence=None, min_height=None):
    """Return indices of local maxima in ``y`` sorted by wavelength.

    Uses SciPy when available, otherwise a small NumPy fallback.
    """
    y = np.asarray(y, dtype=float)
    if prominence is None:
        span = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
        prominence = max(1e-3, 0.02 * span)

    try:
        from scipy.signal import find_peaks as _sp_find

        idx, _ = _sp_find(y, prominence=prominence, height=min_height)
    except Exception:
        idx = []
        for i in range(1, y.size - 1):
            if y[i] > y[i - 1] and y[i] >= y[i + 1]:
                if min_height is None or y[i] >= min_height:
                    left = max(0, i - 1)
                    if y[i] - min(y[left], y[i + 1]) >= prominence:
                        idx.append(i)
        idx = np.asarray(idx, dtype=int)

    order = np.argsort(np.asarray(wl)[idx])
    return np.asarray(idx, dtype=int)[order]


def _fwhm(wl, y, i, baseline):
    """Full width at half maximum of the peak at index ``i`` (linear interp)."""
    y = np.asarray(y, dtype=float)
    wl = np.asarray(wl, dtype=float)
    half = baseline + 0.5 * (y[i] - baseline)

    left = i
    while left > 0 and y[left] > half:
        left -= 1
    right = i
    while right < y.size - 1 and y[right] > half:
        right += 1
    if left == i and right == i:
        return 0.0

    def _cross(i0, i1):
        y0, y1 = y[i0], y[i1]
        if y0 == y1:
            return wl[i0]
        t = (half - y0) / (y1 - y0)
        return wl[i0] + t * (wl[i1] - wl[i0])

    wl_l = _cross(left, left + 1) if left < i else wl[i]
    wl_r = _cross(right - 1, right) if right > i else wl[i]
    return float(abs(wl_r - wl_l))


def resonance_metrics(wl, y_ref, y_test, *, prominence=None):
    """Match peaks and report position / linewidth discrepancies.

    Returns a dict with ``n_ref``, ``n_test``, ``n_matched``,
    ``mean_dwl`` (µm), ``mean_dwidth`` (µm), and the matched pairs.
    """
    wl = np.asarray(wl, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    i_ref = find_peaks(wl, y_ref, prominence=prominence)
    i_test = find_peaks(wl, y_test, prominence=prominence)

    if i_ref.size == 0 or i_test.size == 0:
        return {"n_ref": int(i_ref.size), "n_test": int(i_test.size),
                "n_matched": 0, "mean_dwl": float("nan"),
                "mean_dwidth": float("nan"), "pairs": []}

    wl_ref = wl[i_ref]
    wl_test = wl[i_test]
    span = float(wl.max() - wl.min())
    tol = max(1e-6, 0.1 * span)

    base_ref = float(np.min(y_ref))
    base_test = float(np.min(y_test))

    pairs = []
    used = set()
    for k, w in enumerate(wl_ref):
        dist = np.abs(wl_test - w)
        j = int(np.argmin(dist))
        if dist[j] <= tol and j not in used:
            used.add(j)
            width_ref = _fwhm(wl, y_ref, i_ref[k], base_ref)
            width_test = _fwhm(wl, y_test, i_test[j], base_test)
            pairs.append({
                "wl_ref": float(w), "wl_test": float(wl_test[j]),
                "dwl": float(abs(wl_test[j] - w)),
                "width_ref": width_ref, "width_test": width_test,
                "dwidth": float(abs(width_test - width_ref)),
            })

    if not pairs:
        return {"n_ref": int(i_ref.size), "n_test": int(i_test.size),
                "n_matched": 0, "mean_dwl": float("nan"),
                "mean_dwidth": float("nan"), "pairs": []}

    return {
        "n_ref": int(i_ref.size),
        "n_test": int(i_test.size),
        "n_matched": len(pairs),
        "mean_dwl": float(np.mean([p["dwl"] for p in pairs])),
        "mean_dwidth": float(np.mean([p["dwidth"] for p in pairs])),
        "pairs": pairs,
    }


def compare_spectra(reference: dict, test: dict, *, prominence=None) -> dict:
    """Compare de-embedded RCWA spectra against a de-embedded reference.

    Both dicts contain ``wl_um``, and de-embedded ``R``, ``T``, ``A`` arrays.
    The test spectrum is interpolated onto the reference wavelength grid.
    """
    wl = np.asarray(reference["wl_um"], dtype=float)
    out = {}
    for key in ("R", "T", "A"):
        y_ref = np.asarray(reference[key], dtype=float)
        y_test = interpolate_to(test["wl_um"], test[key], wl)
        out[key] = {
            "rmse": spectral_rmse(y_ref, y_test),
            "max_abs": spectral_max_abs(y_ref, y_test),
        }

    res = resonance_metrics(
        wl, reference["A"], interpolate_to(test["wl_um"], test["A"], wl),
        prominence=prominence,
    )
    out["resonance"] = res
    return out
