import numpy as np
import pytest

meep = pytest.importorskip("meep")

from grating import GratingSpec
from meep_grating import run_grating


def test_lossless_finite_grating_energy_conservation():
    """A lossless dielectric grating must conserve energy: R + T = 1."""
    spec = GratingSpec(
        n_ridges=4, period=1.0, amplitude=0.05, base_thickness=0.03,
        substrate_index=1.5,
    )
    tin = meep.Medium(index=2.0)  # lossless stand-in for TiN
    res = run_grating(
        spec, wl_range=(0.7, 0.95), n_freq=8, angle_deg=0.0, pol="s",
        resolution=30, tin_material=tin, decay_by=1e-6,
    )
    assert np.all(res.R >= -0.01)
    assert np.all(res.T >= -0.01)
    assert np.all(res.R <= 1.01)
    assert np.all(res.T <= 1.01)
    assert np.max(np.abs(res.R + res.T - 1.0)) < 0.03


def test_finite_grating_runs_with_defects():
    spec = GratingSpec(
        n_ridges=5, period=1.0, amplitude=0.05, base_thickness=0.03,
    ).with_defects(jitter_std=0.15, n_missing=1, rng=np.random.default_rng(0))
    tin = meep.Medium(index=2.0)
    res = run_grating(
        spec, wl_range=(0.75, 0.9), n_freq=4, angle_deg=0.0, pol="p",
        resolution=30, tin_material=tin, decay_by=1e-6,
    )
    assert res.R.shape == res.T.shape == res.wl_um.shape
    assert np.all(np.isfinite(res.R)) and np.all(np.isfinite(res.T))
