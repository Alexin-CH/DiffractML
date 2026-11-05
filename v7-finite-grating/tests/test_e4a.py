import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torcwa")

from rcwa_diff import solve_unit_cell  # noqa: E402


def _solve(fill, device="cpu"):
    return solve_unit_cell(
        torch.tensor(1.3, device=device), torch.tensor(0.05, device=device),
        torch.tensor(0.03, device=device), 0.9,
        nh=8, nx=120, n_layers=12, edge_sharpness=100.0, fill=fill,
        device=device,
    )


def test_fill_reduces_coverage():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R_full, T_full = _solve(None, dev)
    R_gap, T_gap = _solve(0.80, dev)
    assert torch.isfinite(R_gap) and torch.isfinite(T_gap)
    # a gap changes the response
    assert abs(float(R_gap) - float(R_full)) > 1e-4


def test_fill_is_differentiable():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fill = torch.tensor(0.85, device=dev, requires_grad=True)
    R, T = _solve(fill, dev)
    (R + T).backward()
    assert fill.grad is not None
    assert torch.isfinite(fill.grad)
    assert fill.grad.abs() > 0


def test_constrained_bounds():
    from e4a_oracle_config import _constrained

    init = torch.tensor(2.0)
    assert math.isclose(float(_constrained(torch.tensor(0.0), init, 0.6, 1.6)), 2.0)
    assert math.isclose(float(_constrained(torch.tensor(10.0), init, 0.6, 1.6)),
                        3.2, rel_tol=1e-6)
    assert math.isclose(float(_constrained(torch.tensor(-10.0), init, 0.6, 1.6)),
                        1.2, rel_tol=1e-6)
