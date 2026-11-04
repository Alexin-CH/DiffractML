import numpy as np

from m2_data import FEATURE_NAMES, build_example, build_features, make_spec


def test_make_spec_deterministic():
    a = make_spec(3)
    b = make_spec(3)
    assert a.period == b.period
    assert np.array_equal(a.ridge_scale, b.ridge_scale)
    assert np.array_equal(a.missing, b.missing)
    assert a.n_ridges == 5


def test_feature_shape_and_names():
    spec = make_spec(0)
    wl = np.linspace(0.7, 1.1, 7)
    R = np.linspace(0.1, 0.2, 7)
    T = np.linspace(0.7, 0.6, 7)
    A = 1 - R - T
    X = build_features(spec, wl, R, T, A)
    assert X.shape == (7, len(FEATURE_NAMES))
    assert "rcwa_A" in FEATURE_NAMES


def test_build_example_residual_consistency():
    spec = make_spec(1)
    wl = np.linspace(0.7, 1.1, 5)
    rcwa = {"R": np.array([0.2, 0.21, 0.22, 0.23, 0.24]),
            "T": np.array([0.6, 0.61, 0.62, 0.63, 0.64]),
            "A": np.array([0.2, 0.18, 0.16, 0.14, 0.12])}
    meep = {"R": rcwa["R"] + 0.01, "T": rcwa["T"] - 0.02,
            "A": rcwa["A"] + 0.01}
    X, Yr, Ya, Raw = build_example(spec, wl, rcwa, meep)
    assert X.shape[0] == 5
    assert Yr.shape == (5, 2)
    assert Ya.shape == (5, 3)
    assert np.allclose(Ya[:, :2] - Raw[:, :2], Yr)
    assert np.allclose(Ya.sum(1), 1.0, atol=1e-6)
