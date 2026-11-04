import numpy as np
import pytest

from tin_dispersion import (
    epsilon_model,
    fit_drude_lorentz,
    load_nk,
    nk_to_epsilon,
)

scipy = pytest.importorskip("scipy")


def test_load_labo_units_and_order():
    wl, n, k = load_nk("labo")
    assert wl.ndim == n.ndim == k.ndim == 1
    assert wl.size == n.size == k.size
    assert np.all(np.diff(wl) > 0)  # ascending
    # labo is tabulated in nm -> converted to µm
    assert 0.2 < wl.min() < 1.0
    assert 0.5 < wl.max() < 2.0
    assert np.all(n > 0) and np.all(k >= 0)


def test_load_beliaev_units():
    wl, n, k = load_nk("beliaev")
    assert np.all(np.diff(wl) > 0)
    assert wl.min() < 0.5 and wl.max() > 2.0


def test_nk_to_epsilon():
    eps = nk_to_epsilon(np.array([1.5]), np.array([0.0]))
    assert np.isclose(eps[0], 2.25 + 0j)
    eps = nk_to_epsilon(np.array([2.0]), np.array([1.0]))
    assert np.isclose(eps[0], 3.0 + 4.0j)


def test_fit_reproduces_labo_band():
    wl, n, k = load_nk("labo")
    params = fit_drude_lorentz(
        wl, n, k, n_osc=3, with_drude=True, wl_range=(0.4, 0.9),
        n_restarts=1, seed=0,
    )
    fit = params["fit"]
    assert np.isfinite(fit["n_rel_rmse"])
    assert np.isfinite(fit["k_rel_rmse"])
    # loose sanity bound: a 3-oscillator fit over a 0.4-0.9 µm band
    assert fit["n_rel_rmse"] < 0.2
    assert fit["k_rel_rmse"] < 0.3


def test_epsilon_model_is_finite():
    wl, n, k = load_nk("labo")
    params = fit_drude_lorentz(
        wl, n, k, n_osc=2, with_drude=False, wl_range=(0.5, 0.9),
        n_restarts=1, seed=0,
    )
    f = np.linspace(1 / 0.9, 1 / 0.5, 20)
    eps = epsilon_model(params, f)
    assert np.all(np.isfinite(eps))
    assert eps.shape == f.shape
