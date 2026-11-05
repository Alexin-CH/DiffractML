"""Differentiable ideal-unit-cell RCWA for Experiment 2.

One clean period, differentiable w.r.t. the effective geometry (period,
amplitude, base) and the edge sharpness.  The encoders in Experiment 2 predict
these values, so gradients must reach them through TORCWA's eigendecomposition.

The grid size is fixed (``nx``) so that ``Lx`` can stay a tensor: only the
sampling positions ``x_i = (i + 0.5) * Lx / nx`` and the k-vectors depend on
``Lx``, both differentiably.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIT_DIR = os.path.normpath(os.path.join(_CURRENT_DIR, "..", "data"))


def tin_epsilon_tensor(wl_um: float, *, tin_params, device, dtype):
    import torch
    from tin_dispersion import epsilon_model

    f = 1.0 / float(wl_um)
    eps = complex(epsilon_model(tin_params, np.array([f]))[0])
    return torch.tensor(eps, dtype=dtype, device=device)


def _propagating_orders(sim):
    import torch

    orders = [
        [int(ox), int(oy)]
        for ox in sim.order_x.tolist()
        for oy in sim.order_y.tolist()
    ]
    m = sim._matching_indices(torch.tensor(orders))

    def _prop(eps_mu):
        kz = torch.sqrt(eps_mu - sim.Kx_norm_dn[m] ** 2 - sim.Ky_norm_dn[m] ** 2)
        return (torch.real(kz) > 1e-6) & (torch.abs(torch.imag(kz)) < 1e-6)

    prop = _prop(sim.eps_in * sim.mu_in) & _prop(sim.eps_out * sim.mu_out)
    return [o for o, p in zip(orders, prop.tolist()) if p]


def _power(sim, direction, port, pol):
    import torch

    orders = _propagating_orders(sim)
    s = sim.s_parameters(
        orders=orders, direction=direction, port=port, polarization=pol,
        ref_order=[0, 0], power_norm=True, evanscent=1e-3,
    )
    return torch.sum(torch.abs(s) ** 2)


def solve_unit_cell(
    period_um,
    amplitude_um,
    base_um,
    wl_um: float,
    *,
    nh: int = 8,
    nx: int = 200,
    n_layers: int = 16,
    edge_sharpness=None,
    fill=None,
    pol: str = "s",
    substrate_index: float = 1.5,
    device="cuda",
    dtype=None,
    tin_params=None,
    tin_dataset: str = "beliaev",
):
    """Differentiable R/T of one ideal period.

    ``period_um``, ``amplitude_um``, ``base_um`` and ``edge_sharpness`` may be
    tensors requiring grad.  ``fill`` (optional) is a duty-cycle fraction in
    ``(0, 1]``: the TiN profile is windowed to a fraction ``fill`` of the
    period, leaving bare substrate elsewhere.  ``fill=1`` reproduces the full
    ideal cell up to a small (1-2 grid point) soft edge at the cell boundary.
    Returns ``(R, T)`` tensors.  ``pol="s"`` maps to y-polarised (TORCWA ``yy``).
    """
    import torch
    import torcwa

    if dtype is None:
        dtype = torch.float32
    if tin_params is None:
        from tin_dispersion import get_or_fit

        tin_params = get_or_fit(
            tin_dataset, cache_path=os.path.join(_FIT_DIR, f"tin_fit_{tin_dataset}.json")
        )
    if edge_sharpness is None:
        edge_sharpness = 100.0
    if not torch.is_tensor(edge_sharpness):
        edge_sharpness = torch.tensor(edge_sharpness, dtype=dtype, device=device)

    Lx_nm = period_um * 1000.0

    torcwa.rcwa_geo.dtype = dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = Lx_nm
    torcwa.rcwa_geo.Ly = Lx_nm
    torcwa.rcwa_geo.nx = int(nx)
    torcwa.rcwa_geo.ny = 1
    torcwa.rcwa_geo.grid()

    X = torcwa.rcwa_geo.x_grid  # (nx, 1), nm
    x_um = (X - Lx_nm / 2.0) / 1000.0
    h_nm = (
        base_um + amplitude_um * (1.0 + torch.sin(2.0 * math.pi * x_um / period_um))
    ) * 1000.0
    if fill is not None:
        # Duty-cycle window: TiN only inside |x| <= fill*period/2, with a
        # 25 nm sigmoid edge.  The +3*edge offset keeps the window open at the
        # cell boundary when fill=1, so the full ideal cell is recovered.
        edge_um = 1.0 / 40.0
        half_um = 0.5 * fill * period_um + 3.0 * edge_um
        mask = torch.sigmoid((half_um - torch.abs(x_um)) / edge_um)
        h_nm = h_nm * mask

    eps_air = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device)
    eps_sub = torch.tensor(
        float(substrate_index) ** 2 + 0.0j, dtype=torch.complex64, device=device
    )
    eps_tin = tin_epsilon_tensor(wl_um, tin_params=tin_params, device=device,
                                 dtype=torch.complex64)

    wl_nm = float(wl_um) * 1000.0
    sim = torcwa.rcwa(
        freq=1.0 / wl_nm, order=[int(nh), 0], lattice=[Lx_nm, Lx_nm],
        dtype=torch.complex64, device=device,
    )
    sim.add_input_layer(eps=eps_air)
    sim.add_output_layer(eps=eps_sub)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")

    H_nm = torch.clamp(h_nm.max(), min=1e-6)
    dz = H_nm / n_layers
    for k in range(n_layers):
        z_mid = H_nm - (k + 0.5) * dz
        mask = torch.sigmoid(edge_sharpness * (h_nm - z_mid))
        eps = mask * eps_tin + (1.0 - mask) * eps_air
        sim.add_layer(thickness=dz, eps=eps)

    sim.solve()
    R = _power(sim, "forward", "reflection", "yy" if pol == "s" else "xx")
    T = _power(sim, "forward", "transmission", "yy" if pol == "s" else "xx")
    return R, T
