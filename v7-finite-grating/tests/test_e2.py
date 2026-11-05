import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torcwa")

from m2_data import geometry_descriptors, make_spec  # noqa: E402
from models_e2 import GeometryEncoder, ResidualHead  # noqa: E402
from rcwa_diff import solve_unit_cell  # noqa: E402


def test_geometry_descriptors():
    spec = make_spec(2)
    d = geometry_descriptors(spec)
    assert d.shape == (8,)
    assert np.all(np.isfinite(d))


def test_encoder_residual_shapes():
    enc = GeometryEncoder(in_dim=8, hidden=16, latent=8)
    z, cfg = enc(torch.zeros(1, 8))
    assert z.shape == (1, 8)
    assert cfg.shape == (1, 2)

    res = ResidualHead(in_dim=8 + 3 + 5 + 8, hidden=16, depth=3)
    assert res(torch.zeros(1, 24)).shape == (1, 2)


def test_rcwa_diff_amplitude_gradient():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def rt(p, a):
        return solve_unit_cell(
            p, a, torch.tensor(0.03, device=dev), 0.9,
            nh=8, nx=120, n_layers=12, edge_sharpness=30.0, device=dev,
        )

    p = torch.tensor(1.3, device=dev, requires_grad=True)
    a = torch.tensor(0.055, device=dev, requires_grad=True)
    R, T = rt(p, a)
    (R + T).sum().backward()

    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert a.grad is not None and torch.isfinite(a.grad).all()

    h = 1e-4
    with torch.no_grad():
        R1, T1 = rt(torch.tensor(1.3, device=dev), torch.tensor(0.055 + h, device=dev))
        R0, T0 = rt(torch.tensor(1.3, device=dev), torch.tensor(0.055 - h, device=dev))
    fd = float(((R1 + T1) - (R0 + T0)) / (2 * h))
    auto = float(a.grad)
    assert fd != 0.0
    # the sharp-mask relaxation makes this approximate; require the right
    # direction and order of magnitude
    assert 0.4 < auto / fd < 2.5
