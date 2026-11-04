import numpy as np

from metrics import (
    bare_substrate,
    compare_spectra,
    deembed,
    find_peaks,
    interpolate_to,
    resonance_metrics,
    spectral_rmse,
)


def test_bare_substrate_values():
    R0, T0 = bare_substrate(0.0, "s")
    assert abs(R0 - 0.04) < 1e-9
    assert abs(T0 - 0.96) < 1e-9


def test_deembed_recovers_aperture():
    # Area-weighted mixture: padding bare substrate + aperture grating.
    W, Lx = 10.0, 15.0
    R0, T0 = 0.04, 0.96
    R_ap, T_ap = 0.5, 0.3
    frac = W / Lx
    R_total = R_ap * frac + R0 * (1 - frac)
    T_total = T_ap * frac + T0 * (1 - frac)
    R, T, A = deembed([R_total], [T_total], Lx_um=Lx, W_um=W, R0=R0, T0=T0)
    assert np.isclose(R[0], R_ap)
    assert np.isclose(T[0], T_ap)
    assert np.isclose(A[0], 1 - R_ap - T_ap)
    assert np.isclose(R[0] + T[0] + A[0], 1.0)


def test_deembed_absorption_is_baseline_free():
    # The bare substrate absorbs nothing, so A*(Lx/W) is exact.
    W, Lx = 8.0, 12.0
    A_total = 0.2 * (W / Lx)
    _, _, A = deembed([0.0], [1.0 - A_total], Lx_um=Lx, W_um=W, R0=0.04, T0=0.96)
    assert np.isclose(A[0], 0.2)


def test_spectral_rmse_identity():
    y = np.array([1.0, 2.0, 3.0])
    assert spectral_rmse(y, y) == 0.0
    assert np.isclose(spectral_rmse(y, y + 1), 1.0)


def test_interpolate_to_unordered_source():
    wl = np.array([1.0, 0.5, 1.5])
    y = np.array([10.0, 5.0, 15.0])
    out = interpolate_to(wl, y, [0.75, 1.25])
    assert np.allclose(out, [7.5, 12.5])


def test_find_peaks_and_resonance_metrics():
    wl = np.linspace(0.5, 1.5, 401)
    y_ref = np.exp(-((wl - 0.8) ** 2) / 0.0005) + np.exp(-((wl - 1.2) ** 2) / 0.0005)
    y_test = np.exp(-((wl - 0.805) ** 2) / 0.0005) + np.exp(-((wl - 1.2) ** 2) / 0.0005)
    peaks = find_peaks(wl, y_ref)
    assert peaks.size == 2
    res = resonance_metrics(wl, y_ref, y_test)
    assert res["n_matched"] == 2
    assert res["mean_dwl"] < 0.01


def test_compare_spectra_self_is_zero():
    wl = np.linspace(0.6, 1.0, 50)
    spec = {"wl_um": wl, "R": wl * 0.1, "T": 1 - wl * 0.1, "A": np.zeros_like(wl)}
    out = compare_spectra(spec, spec)
    assert out["R"]["rmse"] == 0.0
    assert out["A"]["rmse"] == 0.0
