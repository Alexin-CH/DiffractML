import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torcwa")

from rcwa_diff import solve_unit_cell  # noqa: E402
from rcwa_supercell_diff import profile_from_controls, solve_profile  # noqa: E402


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_eps_scale_changes_spectrum():
    d = _dev()
    args = (torch.tensor(1.3, device=d), torch.tensor(0.05, device=d),
            torch.tensor(0.03, device=d), 0.9)
    R1, T1 = solve_unit_cell(*args, nh=8, nx=120, n_layers=12,
                             eps_scale=1.0, device=d)
    R2, T2 = solve_unit_cell(*args, nh=8, nx=120, n_layers=12,
                             eps_scale=0.6, device=d)
    assert torch.isfinite(R2) and torch.isfinite(T2)
    assert abs(float(R1) - float(R2)) + abs(float(T1) - float(T2)) > 1e-4


def test_eps_scale_differentiable():
    d = _dev()
    s = torch.tensor(1.0, device=d, requires_grad=True)
    R, T = solve_unit_cell(torch.tensor(1.3, device=d),
                           torch.tensor(0.05, device=d),
                           torch.tensor(0.03, device=d), 0.9,
                           nh=8, nx=120, n_layers=12, eps_scale=s, device=d)
    (R + T).backward()
    assert s.grad is not None and torch.isfinite(s.grad)
    assert s.grad.abs() > 0


def test_profile_from_controls_shape():
    k, nx = 8, 64
    ctrl = torch.full((k,), 0.05)
    prof = profile_from_controls(ctrl, nx, 4.0)
    assert prof.shape == (nx,)
    assert torch.allclose(prof, torch.full((nx,), 0.05), atol=1e-6)


def test_solve_profile_runs():
    d = _dev()
    nx = 96
    Lx = 4.0
    flat = torch.full((nx,), 0.04, device=d)
    R, T = solve_profile(flat, Lx, 0.9, nh=6, nx=nx, n_layers=10, device=d)
    assert torch.isfinite(R) and torch.isfinite(T)
    assert 0.0 <= float(R) <= 1.0 and 0.0 <= float(T) <= 1.0

    prof = torch.zeros((nx,), device=d)
    prof[nx // 3: 2 * nx // 3] = 0.08
    R2, T2 = solve_profile(prof, Lx, 0.9, nh=6, nx=nx, n_layers=10, device=d)
    assert abs(float(R) - float(R2)) + abs(float(T) - float(T2)) > 1e-4
