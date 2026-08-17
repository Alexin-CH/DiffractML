"""Regression tests for the corrected TORCWA wrapper."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation import (  # noqa: E402
    S_PARAM_KEYS,
    SinTinArgs,
    get_power_efficiencies,
    get_s_parameters,
    setup,
)


def make_args(**kw):
    base = dict(
        wl=1000.0,
        ang=30.0,
        nh=8,
        discretization=48,
        sin_amplitude=45.0,
        sin_period=1500.0,
        uni_layer_h=30.0,
    )
    base.update(kw)
    return SinTinArgs(**base)


def test_s_params_are_distinct():
    """The original bug: all 32 S-parameter columns were identical."""
    sim = setup(make_args())
    params = get_s_parameters(sim)
    assert len(params) == len(S_PARAM_KEYS)
    vals = [abs(params[k].item()) for k in S_PARAM_KEYS]
    assert len(set(round(v, 4) for v in vals)) > 4


def test_power_conservation_with_absorption():
    """R_xx + T_xx < 1 (and likewise for yy) for absorbing TiN."""
    sim = setup(make_args())
    eff = get_power_efficiencies(sim)
    for pol in ("xx", "yy"):
        total = eff[f"R_{pol}"].item() + eff[f"T_{pol}"].item()
        assert 0.5 < total < 1.0, f"R_{pol}+T_{pol}={total} out of (0.5, 1.0)"
    for k in eff:
        v = eff[k].item()
        assert 0.0 <= v <= 1.0
        assert torch.isfinite(eff[k])


def test_cross_polarization_is_zero():
    """1D grating: cross-polarized channels are ~0."""
    sim = setup(make_args())
    sxx = sim.s_parameters(
        orders=[[0, 0]], direction="forward", port="reflection",
        polarization="yx", ref_order=[0, 0], power_norm=True,
    )
    assert abs(sxx.item()) < 1e-3


def test_amplitude_gradient_flows():
    """amp gradient must be nonzero through the solver (smooth masks)."""
    args = make_args(sin_amplitude=torch.tensor(45.0, requires_grad=True))
    sim = setup(args)
    eff = get_power_efficiencies(sim)
    loss = eff["R_xx"]
    loss.backward()
    g = args.sin_amplitude.grad.item()
    assert torch.isfinite(torch.tensor(g))
    assert abs(g) > 1e-5


def test_response_changes_with_amplitude():
    """Physical sanity: different amp -> different R_xx."""
    r1 = get_power_efficiencies(setup(make_args(sin_amplitude=30.0)))["R_xx"]
    r2 = get_power_efficiencies(setup(make_args(sin_amplitude=60.0)))["R_xx"]
    assert abs(r1.item() - r2.item()) > 0.01