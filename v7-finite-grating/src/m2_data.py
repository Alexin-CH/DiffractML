"""Shared definitions for M2: paired (fast RCWA, MEEP) spectra + features.

M2 is the learned spectral correction on top of the de-embedded padded-supercell
RCWA result.  For each (grating, wavelength) we build a feature vector from the
geometry + illumination + the fast RCWA spectrum, and regress the *residual*
between the MEEP reference and RCWA.

The residual is taken on ``(R, T)`` only; ``A = 1 - R - T`` is derived, so the
predicted spectrum automatically satisfies ``R + T + A = 1``.
"""

from __future__ import annotations

import numpy as np

from grating import GratingSpec, sample_grating_spec

N_RIDGES = 5
WL_DEFAULT = (0.7, 1.15)


def reference_base(idx: int, pol: str, angle: float,
                   spec_set: str = "m2") -> str:
    """Filename stem shared by the generator and the dataset builder."""
    if spec_set == "m2" and abs(angle) < 1e-9:
        return f"m2_{idx:03d}_{pol}"
    tag = "m2" if spec_set == "m2" else "res"
    return f"{tag}_{idx:03d}_{pol}_a{angle:02.0f}"

# The "fast" RCWA representation used as the model input.
RCWA_CFG = dict(padding_um=1.0, nh=32, grid_nm=10.0, n_layers=40)

_GEOM_SCALARS = ["period", "amplitude", "base"]
_DEFECT = ["mean_scale", "std_scale", "min_scale", "n_missing", "missing_index"]
_DERIVED = ["wl", "inv_wl", "wl_over_period", "amp_over_period", "base_over_period"]
_RCWA = ["rcwa_R", "rcwa_T", "rcwa_A"]
FEATURE_NAMES = _GEOM_SCALARS + _DEFECT + _DERIVED + _RCWA


def make_spec(idx: int) -> GratingSpec:
    """Deterministic grating spec for dataset index ``idx`` (shared by stages)."""
    rng = np.random.default_rng(1000 + int(idx))
    return sample_grating_spec(
        rng,
        n_ridges=N_RIDGES,
        period_range=(1.0, 1.7),
        amplitude_range=(0.03, 0.07),
        base_thickness=0.03,
        jitter_std=0.1,
        max_missing=1,
    )


# --------------------------------------------------------------------------- #
# Experiment 4d: Rayleigh-rich finite gratings (small period, many ridges)
# --------------------------------------------------------------------------- #
RESONANT_N_RIDGES = 10


def make_resonant_spec(idx: int) -> GratingSpec:
    """Small-period, 10-ridge grating whose Rayleigh anomalies fall in band.

    With ``p in [0.4, 0.8] um`` and 30-60 deg incidence, ``p(1+sinθ)`` and
    ``n_sub·p(1+sinθ)`` cross 0.7-1.15 um.  No missing ridge, so the anomaly
    stays the dominant spectral feature.
    """
    rng = np.random.default_rng(5000 + int(idx))
    return sample_grating_spec(
        rng,
        n_ridges=RESONANT_N_RIDGES,
        period_range=(0.40, 0.80),
        amplitude_range=(0.040, 0.060),
        base_thickness=0.03,
        jitter_std=0.1,
        max_missing=0,
    )


def build_features(spec: GratingSpec, wl, rcwa_R, rcwa_T, rcwa_A) -> np.ndarray:
    """Feature matrix ``(n_wl, len(FEATURE_NAMES))`` for one grating."""
    wl = np.asarray(wl, dtype=np.float64)
    scale = np.asarray(spec.ridge_scale, dtype=float)
    miss = np.asarray(spec.missing, dtype=bool)
    n_missing = int(miss.sum())
    missing_index = int(np.argmax(miss)) - N_RIDGES // 2 if n_missing > 0 else -99
    rows = []
    for i, w in enumerate(wl):
        rows.append([
            spec.period, spec.amplitude, spec.base_thickness,
            scale.mean(), scale.std(), scale.min(),
            n_missing, missing_index,
            w, 1.0 / w, w / spec.period,
            spec.amplitude / spec.period, spec.base_thickness / spec.period,
            rcwa_R[i], rcwa_T[i], rcwa_A[i],
        ])
    x = np.asarray(rows, dtype=np.float32)
    assert x.shape[1] == len(FEATURE_NAMES), (x.shape, len(FEATURE_NAMES))
    return x


def build_example(spec, wl, rcwa, meep):
    """Return ``(X, Y_residual, Y_abs, RCWA_abs)`` for one grating.

    ``rcwa`` / ``meep`` are dicts with de-embedded ``R, T, A`` arrays.
    ``Y_residual`` has columns ``[dR, dT]``; ``Y_abs`` has ``[R, T, A]``.
    """
    X = build_features(spec, wl, rcwa["R"], rcwa["T"], rcwa["A"])
    Y_res = np.column_stack([
        meep["R"] - rcwa["R"],
        meep["T"] - rcwa["T"],
    ]).astype(np.float32)
    Y_abs = np.column_stack([meep["R"], meep["T"], meep["A"]]).astype(np.float32)
    RCWA_abs = np.column_stack([rcwa["R"], rcwa["T"], rcwa["A"]]).astype(np.float32)
    return X, Y_res, Y_abs, RCWA_abs


# --------------------------------------------------------------------------- #
# Experiment 2: compact geometry descriptors for the encoder
# --------------------------------------------------------------------------- #
GEOM_DESCRIPTOR_NAMES = [
    "period_s", "amplitude_s", "base_s",
    "mean_scale", "std_scale", "min_scale",
    "n_missing", "missing_index_s",
]


def geometry_descriptors(spec: GratingSpec) -> np.ndarray:
    """Scale-invariant 8-d description of a grating (for the E2 encoder)."""
    scale = np.asarray(spec.ridge_scale, dtype=float)
    miss = np.asarray(spec.missing, dtype=bool)
    n_missing = int(miss.sum())
    missing_index = (int(np.argmax(miss)) - N_RIDGES // 2) / max(1, N_RIDGES) \
        if n_missing > 0 else -1.0
    return np.asarray([
        spec.period / 1.5,
        spec.amplitude / 0.05,
        spec.base_thickness / 0.03,
        scale.mean(), scale.std(), scale.min(),
        float(n_missing), missing_index,
    ], dtype=np.float32)
