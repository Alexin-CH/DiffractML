import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torcwa")

from models_e3 import ConfigHead, GlobalEncoder, RidgeSetEncoder  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "..", "reports", "m2",
                     "dataset_ideal_nh8.npz")


def test_global_encoder_shapes():
    enc = GlobalEncoder(8, hidden=16, latent=8)
    z = enc(torch.zeros(3, 8))
    assert z.shape == (3, 8)


def test_ridge_encoder_shapes_and_pooling():
    enc = RidgeSetEncoder(3, 5, hidden=16, latent=8)
    tok = torch.zeros(2, 5, 3)
    glob = torch.zeros(2, 5)
    z = enc(tok, glob)
    assert z.shape == (2, 8)

    # mean/max pooling over ridge tokens is permutation-invariant
    tok2 = torch.stack([tok[0][[4, 2, 0, 3, 1]], tok[1][[1, 0, 4, 2, 3]]])
    z2 = enc(tok2, glob)
    assert torch.allclose(z, z2, atol=1e-6)


def test_config_head_bounds():
    z = torch.randn(64, 4)
    head = ConfigHead(4, "amp_period")
    la, lp = head(z)
    assert la is not None and lp is not None
    assert la.abs().max() <= 0.15 + 1e-6
    assert lp.abs().max() <= 0.03 + 1e-6

    head_none = ConfigHead(4, "none")
    assert head_none(z) == (None, None)


@pytest.mark.skipif(not os.path.exists(_DATA), reason="E3 dataset missing")
def test_e3_static_features_and_split():
    import train_e3 as e3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = e3.E3Data(_DATA, torch.device(dev))
    assert data.static.shape == (len(data.group), 13)
    assert data.tokens.shape[1:] == (5, 3)
    assert data.glob.shape[1] == 5

    tr, va = e3.split_ids(data.ids, 0.25, 0)
    assert len(va) == 6 and len(tr) == len(data.ids) - 6

    shifted = data._static(amp_scale=1.1)
    # amplification shifts the amp column and the amp/period ratio
    assert not torch.allclose(shifted, data.static)


@pytest.mark.skipif(not os.path.exists(_DATA), reason="E3 dataset missing")
def test_e3_in_loop_gradient_reaches_amplitude():
    import train_e3 as e3
    from tin_dispersion import get_or_fit

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = e3.E3Data(_DATA, torch.device(dev))
    tin = get_or_fit("beliaev",
                     cache_path=os.path.join(
                         os.path.dirname(__file__), "..", "data",
                         "tin_fit_beliaev.json"))

    cfg = e3.parse([])
    cfg.device = str(dev)
    cfg.nh, cfg.nx, cfg.n_layers = 8, 120, 16
    cfg.in_loop = True
    cfg.cfg_mode = "amp"
    cfg.encoder = "global"
    cfg.base_scale = 0.03

    model = e3.HybridModel("global", "amp", latent=8, hidden=16, depth=2).to(dev)
    solve = e3._solve_fn(data, cfg, tin)
    rows = torch.tensor([0, 1], device=dev)
    base = e3._base_features(data, rows)
    norm = (torch.zeros(2, device=dev), torch.ones(2, device=dev),
            base.mean(0), base.std(0).clamp(min=1e-8))

    loss, *_ = e3._loss_and_delta(model, data, rows, norm,
                                  lambda *args, **kw: e3._forward_in_loop(
                                      *args[:3], solve))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    # the config head must receive gradient through the differentiable solve
    cfg_grad = model.cfg.fc.weight.grad
    assert cfg_grad is not None and cfg_grad.abs().sum() > 0
