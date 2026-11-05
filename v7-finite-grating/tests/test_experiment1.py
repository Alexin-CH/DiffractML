import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torcwa")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from benchmark_ideal_vs_supercell import ideal_spec_from  # noqa: E402
from m2_data import make_spec  # noqa: E402


def test_ideal_spec_is_clean_single_period():
    spec = make_spec(3)
    ispec = ideal_spec_from(spec)
    assert ispec.n_ridges == 1
    assert ispec.period == spec.period
    assert ispec.amplitude == spec.amplitude
    assert ispec.base_thickness == spec.base_thickness
    assert ispec.substrate_index == spec.substrate_index
    assert np.allclose(ispec.ridge_scale, 1.0)
    assert not ispec.missing.any()
    # one period wide
    assert np.isclose(ispec.aperture_width, spec.period)
