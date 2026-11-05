import pytest

torch = pytest.importorskip("torch")

from models_m2 import CorrectionMLP  # noqa: E402


def test_forward_shape():
    model = CorrectionMLP(in_dim=21, hidden=32, depth=3, out_dim=2)
    x = torch.randn(10, 21)
    y = model(x)
    assert y.shape == (10, 2)


def test_gradients_flow():
    model = CorrectionMLP(in_dim=8, hidden=16, depth=3, out_dim=2)
    x = torch.randn(4, 8, requires_grad=True)
    loss = model(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
