"""Differentiable padded-supercell RCWA with an arbitrary TiN height profile.

This is the solver behind the learned low-order supercell (Experiment 4c): the
TiN fill is not an analytic sine but a profile ``h(x)`` supplied by the caller
— eventually by a network — and the periodic solve runs at a *fixed low* number
of Fourier orders.

The cell is the finite aperture plus air/substrate padding, exactly like
:mod:`rcwa_supercell`, but ``h(x)`` is a tensor so gradients flow through it.
Everything else (TiN dispersion, layer discretisation, order truncation) is
shared with ``rcwa_diff``.
"""

from __future__ import annotations

import math
import os

import torch

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIT_DIR = os.path.normpath(os.path.join(_CURRENT_DIR, "..", "data"))


def solve_profile(
    height_um,
    Lx_um,
    wl_um: float,
    *,
    nh: int = 8,
    nx: int = 256,
    n_layers: int = 40,
    edge_sharpness: float = 100.0,
    pol: str = "s",
    substrate_index: float = 1.5,
    device="cuda",
    tin_params=None,
    tin_dataset: str = "beliaev",
):
    """RCWA ``(R, T)`` of a periodic profile ``height_um`` (µm) over ``Lx_um``.

    ``height_um`` is a 1-D tensor of TiN thickness sampled on ``nx`` points; it
    may require grad.  The profile is periodic in ``x`` with period ``Lx_um``
    and repeats along ``y`` (1-D grating, invariant along y).
    """
    import torcwa

    from rcwa_diff import _power, tin_epsilon_tensor

    if tin_params is None:
        from tin_dispersion import get_or_fit

        tin_params = get_or_fit(
            tin_dataset,
            cache_path=os.path.join(_FIT_DIR, f"tin_fit_{tin_dataset}.json"),
        )

    device = torch.device(device)
    dtype = torch.float32
    Lx_nm = float(Lx_um) * 1000.0

    torcwa.rcwa_geo.dtype = dtype
    torcwa.rcwa_geo.device = str(device)
    torcwa.rcwa_geo.Lx = Lx_nm
    torcwa.rcwa_geo.Ly = Lx_nm
    torcwa.rcwa_geo.nx = int(nx)
    torcwa.rcwa_geo.ny = 1
    torcwa.rcwa_geo.grid()

    h_nm = height_um.to(device=device, dtype=dtype) * 1000.0
    if h_nm.numel() != int(nx):
        raise ValueError(f"height_um must have {nx} samples, got {h_nm.numel()}")
    h_nm = h_nm.reshape(-1, 1)

    eps_air = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device)
    eps_sub = torch.tensor(
        float(substrate_index) ** 2 + 0.0j, dtype=torch.complex64, device=device
    )
    eps_tin = tin_epsilon_tensor(wl_um, tin_params=tin_params, device=device,
                                 dtype=torch.complex64)

    wl_nm = float(wl_um) * 1000.0
    sim = torcwa.rcwa(
        freq=1.0 / wl_nm, order=[int(nh), 0], lattice=[Lx_nm, Lx_nm],
        dtype=torch.complex64, device=str(device),
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


def profile_from_controls(controls_um, nx: int, Lx_um: float):
    """Interpolate ``K`` control heights to the ``nx`` grid (linear).

    ``controls_um`` may require grad; ``Lx_um`` is the cell width.  Control
    points are cell-centred so the profile stays periodic.
    """
    k = controls_um.numel()
    x_ctrl = (torch.arange(k, device=controls_um.device, dtype=torch.float32)
              + 0.5) * (float(Lx_um) / k)
    x = (torch.arange(nx, device=controls_um.device, dtype=torch.float32)
         + 0.5) * (float(Lx_um) / nx)
    # linear interpolation with periodic wrap
    idx = (x / float(Lx_um) * k - 0.5)
    i0 = torch.floor(idx)
    frac = idx - i0
    i0 = i0.long() % k
    i1 = (i0 + 1) % k
    return controls_um[i0] * (1.0 - frac) + controls_um[i1] * frac


def true_profile(spec, Lx_um: float, nx: int):
    """Sample the real defective grating on the supercell grid (for init)."""
    half = 0.5 * Lx_um
    x = (torch.arange(nx, dtype=torch.float32) + 0.5) * (Lx_um / nx) - half
    h = torch.zeros(nx)
    # aperture occupies the central W = n_ridges*period
    W = spec.n_ridges * spec.period
    centers = spec.ridge_centers
    for c, scale, miss in zip(centers, spec.ridge_scale, spec.missing):
        if miss:
            continue
        local = (x - float(c)) / float(spec.period)
        inside = (x >= float(c) - 0.5 * spec.period) & \
                 (x <= float(c) + 0.5 * spec.period)
        prof = (spec.base_thickness
                + spec.amplitude * scale * (1.0 + torch.sin(2 * math.pi * local)))
        h = torch.where(inside, prof.float(), h)
    del W
    return h
