import numpy as np
import pytest

meep = pytest.importorskip("meep")

from fresnel import fresnel_RT
from grating import GratingSpec
from meep_grating import run_grating

# A grating with zero corrugation / zero base is a planar interface.
PLANAR = GratingSpec(
    n_ridges=1, period=1.0, amplitude=0.0, base_thickness=0.0,
    substrate_index=1.5,
)


def _dummy_tin():
    return meep.Medium(index=1.0)


def test_planar_normal_s_pol():
    wl_range = (0.6, 1.0)
    res = run_grating(
        PLANAR, wl_range=wl_range, n_freq=8, angle_deg=0.0, pol="s",
        resolution=30, tin_material=_dummy_tin(), decay_by=1e-6,
    )
    R_ref, _ = fresnel_RT(1.0, 1.5, 0.0, "s")
    assert np.max(np.abs(res.R - R_ref)) < 0.02
    assert np.max(np.abs(res.R + res.T - 1.0)) < 0.02


def test_planar_normal_p_pol():
    res = run_grating(
        PLANAR, wl_range=(0.6, 1.0), n_freq=8, angle_deg=0.0, pol="p",
        resolution=30, tin_material=_dummy_tin(), decay_by=1e-6,
    )
    R_ref, _ = fresnel_RT(1.0, 1.5, 0.0, "p")
    assert np.max(np.abs(res.R - R_ref)) < 0.02
    assert np.max(np.abs(res.R + res.T - 1.0)) < 0.02


def test_planar_oblique_s_pol():
    angle = 30.0
    res = run_grating(
        PLANAR, wl_range=(0.85, 0.85), n_freq=1, angle_deg=angle, pol="s",
        resolution=50, tin_material=_dummy_tin(), decay_by=1e-8,
        pad=2.0, air_above=1.2,
    )
    R_ref, _ = fresnel_RT(1.0, 1.5, angle, "s")
    assert abs(float(res.R[0]) - R_ref) < 0.01
