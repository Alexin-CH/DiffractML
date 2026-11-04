import numpy as np

from grating import GratingSpec, sample_grating_spec


def test_aperture_and_centers():
    spec = GratingSpec(n_ridges=4, period=2.0, amplitude=0.05)
    assert spec.aperture_width == 8.0
    assert np.allclose(spec.ridge_centers, [-3.0, -1.0, 1.0, 3.0])


def test_height_outside_aperture_is_zero():
    spec = GratingSpec(n_ridges=3, period=1.0, amplitude=0.05, base_thickness=0.03)
    x = np.array([-2.0, -1.6, 1.6, 2.0, 5.0])
    assert np.allclose(spec.height_profile(x), 0.0)


def test_sine_profile_inside_aperture():
    spec = GratingSpec(n_ridges=2, period=1.0, amplitude=0.05, base_thickness=0.03)
    # ridge centres in the aperture are at x = -0.5, 0.5; sin(0)=0 -> base+amp
    centers = spec.ridge_centers
    h_center = spec.height_profile(centers)
    assert np.allclose(h_center, 0.03 + 0.05)
    # quarter period away: sin = +1 -> base + 2*amp, sin = -1 -> base
    h_peak = spec.height_profile(centers + 0.25 * spec.period)
    h_trough = spec.height_profile(centers - 0.25 * spec.period)
    assert np.allclose(h_peak, 0.03 + 2 * 0.05, atol=1e-9)
    assert np.allclose(h_trough, 0.03, atol=1e-9)


def test_missing_ridge_removes_material():
    missing = np.array([False, True, False, False])
    spec = GratingSpec(
        n_ridges=4, period=1.0, amplitude=0.05, base_thickness=0.03,
        missing=missing,
    )
    # ridge centres are -1.5, -0.5, 0.5, 1.5; the second one is missing
    h = spec.height_profile(spec.ridge_centers)
    assert h[1] == 0.0
    assert np.all(h[[0, 2, 3]] > 0.0)


def test_height_jitter_scales_amplitude():
    base = GratingSpec(n_ridges=3, period=1.0, amplitude=0.05, base_thickness=0.0)
    scaled = GratingSpec(
        n_ridges=3, period=1.0, amplitude=0.05, base_thickness=0.0,
        ridge_scale=np.array([1.0, 0.5, 0.0]),
    )
    x = base.ridge_centers  # sin=0 -> corrugation = amplitude * ridge_scale
    h_base = base.height_profile(x)
    h_scaled = scaled.height_profile(x)
    assert np.allclose(h_base, 0.05)
    assert np.allclose(h_scaled, 0.05 * np.array([1.0, 0.5, 0.0]))
    # quarter period: corrugation = amplitude*(1+1), scaled by ridge_scale
    h_peak_scaled = scaled.height_profile(x + 0.25)
    assert np.allclose(h_peak_scaled, 2 * 0.05 * np.array([1.0, 0.5, 0.0]))


def test_defects_reproducible_and_bounded():
    spec = GratingSpec(n_ridges=8, period=1.0, amplitude=0.05)
    a = spec.with_defects(jitter_std=0.2, n_missing=2, rng=np.random.default_rng(0),
                          min_scale=0.1)
    b = spec.with_defects(jitter_std=0.2, n_missing=2, rng=np.random.default_rng(0),
                          min_scale=0.1)
    assert np.array_equal(a.ridge_scale, b.ridge_scale)
    assert np.array_equal(a.missing, b.missing)
    assert a.missing.sum() == 2
    assert np.all(a.ridge_scale >= 0.1)


def test_sample_spec_is_valid():
    rng = np.random.default_rng(1)
    spec = sample_grating_spec(rng, n_ridges=6)
    assert spec.n_ridges == 6
    assert spec.period > 0
    assert spec.height_profile(np.zeros(5)).shape == (5,)
