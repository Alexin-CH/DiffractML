import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torcwa")

from grating import GratingSpec  # noqa: E402
from rcwa_supercell import RCWAConfig, solve_supercell  # noqa: E402


def _tmm_slab(n1, n2, n3, d_um, wl_um):
    b = 2 * np.pi * n2 * d_um / wl_um
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    E = np.exp(2j * b)
    r = (r12 + r23 * E) / (1 + r12 * r23 * E)
    return float(abs(r) ** 2)


def test_bare_substrate_fresnel():
    spec = GratingSpec(n_ridges=1, period=1.0, amplitude=0.0, base_thickness=0.0,
                       substrate_index=1.5)
    cfg = RCWAConfig(padding_um=0.5, nh=8, grid_nm=20.0, device="cpu")
    eff = solve_supercell(spec, 0.9, cfg, eps_override=1.0)
    assert abs(eff["R_xx"] - 0.04) < 1e-3
    assert abs(eff["T_xx"] - 0.96) < 1e-3
    assert abs(eff["R_yy"] - 0.04) < 1e-3


def test_uniform_slab_matches_tmm():
    # A uniform slab filling the lattice (pad=0) must match transfer-matrix.
    spec = GratingSpec(n_ridges=4, period=1.0, amplitude=0.0, base_thickness=0.05,
                       substrate_index=1.5)
    cfg = RCWAConfig(padding_um=0.0, nh=8, grid_nm=20.0, n_layers=60, device="cpu")
    eff = solve_supercell(spec, 0.9, cfg, eps_override=complex(4.0, 0.0))
    R_ref = _tmm_slab(1.0, 2.0, 1.5, 0.05, 0.9)
    assert abs(eff["R_xx"] - R_ref) < 5e-3
    assert abs(eff["T_xx"] - (1 - R_ref)) < 5e-3


def test_supercell_dilutes_and_deembed_recovers():
    # With padding, raw R is diluted; de-embedding should approach the slab R.
    from metrics import deembed

    spec = GratingSpec(n_ridges=6, period=1.0, amplitude=0.0, base_thickness=0.05,
                       substrate_index=1.5)
    cfg = RCWAConfig(padding_um=1.0, nh=8, grid_nm=20.0, n_layers=60, device="cpu")
    eff = solve_supercell(spec, 0.9, cfg, eps_override=complex(4.0, 0.0))
    R_d, _, _ = deembed(eff["R_xx"], eff["T_xx"], Lx_um=eff["Lx_um"],
                        W_um=eff["W_um"], R0=0.04, T0=0.96)
    assert eff["R_xx"] < 0.1  # diluted
    assert abs(float(R_d) - _tmm_slab(1.0, 2.0, 1.5, 0.05, 0.9)) < 0.03
