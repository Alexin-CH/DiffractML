import os

import numpy as np

from benchmark import (
    deembed_rcwa,
    load_reference,
    save_reference,
    spec_from_dict,
    spec_to_dict,
)
from grating import GratingSpec


class _DummyResult:
    def __init__(self, wl, R, T, spec, pol="s", angle=0.0, Lx=10.0):
        self.wl_um = np.asarray(wl, float)
        self.R = np.asarray(R, float)
        self.T = np.asarray(T, float)
        self.pol = pol
        self.angle_deg = angle
        self.meta = {"sx": Lx, "resolution": 150.0}
        self.spec = spec


def test_spec_roundtrip():
    spec = GratingSpec(n_ridges=5, period=1.1, amplitude=0.04, base_thickness=0.03)
    spec = spec.with_defects(jitter_std=0.1, n_missing=1, rng=np.random.default_rng(3))
    d = spec_to_dict(spec)
    back = spec_from_dict(d)
    assert back.n_ridges == spec.n_ridges
    assert np.allclose(back.ridge_scale, spec.ridge_scale)
    assert np.array_equal(back.missing, spec.missing)


def test_save_load_reference_roundtrip(tmp_path):
    spec = GratingSpec(n_ridges=4, period=1.0, amplitude=0.05, base_thickness=0.03)
    wl = np.linspace(0.6, 1.0, 7)
    R = np.linspace(0.1, 0.2, 7)
    T = 1.0 - R - 0.05
    res = _DummyResult(wl, R, T, spec, Lx=9.0)

    base = os.path.join(tmp_path, "ref")
    csv_path, json_path = save_reference(res, base, spec=spec)
    assert os.path.exists(csv_path) and os.path.exists(json_path)

    loaded = load_reference(base)
    assert np.allclose(loaded["wl_um"], wl)
    assert np.allclose(loaded["R"], R)
    assert np.allclose(loaded["T"], T)
    assert loaded["Lx_um"] == 9.0
    assert loaded["spec"].n_ridges == 4


def test_deembed_rcwa_shapes():
    # A bare-substrate total (R0, T0) de-embeds to itself with zero absorption.
    res = {
        "wl_um": np.array([0.7, 0.9]),
        "R": np.array([0.04, 0.04]),
        "T": np.array([0.96, 0.96]),
        "Lx_um": 12.0,
        "W_um": 8.0,
    }
    out = deembed_rcwa(res, 0.0, "s")
    assert np.allclose(out["A"], 0.0)
    assert np.allclose(out["R"], 0.04)
    assert np.allclose(out["T"], 0.96)
    assert "R_raw" in out
