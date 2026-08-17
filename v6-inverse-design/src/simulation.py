"""Correct, GPU-capable, differentiable wrapper around the TORCWA RCWA solver
for the 1D sine-corrugated TiN structure.

Fixes the bugs found in the internship code (archive/v5-hybrid-order):
  * ``S_parameters`` -> ``s_parameters`` (TORCWA method name)
  * direction / port / polarization loop variables were hardcoded (all 32
    S-parameter columns ended up identical)
  * hard-threshold (``h >= z_mid``) masks -> smooth sigmoid masks so that
    gradients flow w.r.t. geometry parameters
"""

import os
import sys

import torch

_TORCWA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "TORCWA")
)
if _TORCWA_ROOT not in sys.path:
    sys.path.insert(0, _TORCWA_ROOT)

import src as torcwa  # noqa: E402

_INDEX_CSV = os.path.join(
    _TORCWA_ROOT, "simulations", "sin_tin", "TiN-RefIdx-labo.csv"
)

POLARIZATIONS = ["xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"]
DIRECTIONS = ["forward", "backward"]
PORTS = ["transmission", "reflection"]

S_PARAM_KEYS = [
    f"{d}.{p}.{pol}" for d in DIRECTIONS for p in PORTS for pol in POLARIZATIONS
]


class SinTinArgs:
    """Simulation parameters for the 1D sine TiN grating.

    ``wl``, ``ang``, ``sin_amplitude`` and ``sin_period`` are kept as
    floating tensors so gradients can flow through the solver.
    """

    def __init__(
        self,
        wl,
        ang,
        nh,
        discretization,
        sin_amplitude,
        sin_period,
        uni_layer_h,
        *,
        edge_sharpness=100.0,
        device="cpu",
        dtype=torch.float32,
    ):
        self.wl = torch.as_tensor(wl, dtype=dtype, device=device).requires_grad_()
        self.ang = torch.as_tensor(ang, dtype=dtype, device=device).requires_grad_()
        self.nh = int(nh)
        self.discretization = int(discretization)
        self.sin_amplitude = torch.as_tensor(
            sin_amplitude, dtype=dtype, device=device
        ).requires_grad_()
        self.sin_period = torch.as_tensor(
            sin_period, dtype=dtype, device=device
        ).requires_grad_()
        self.uni_layer_h = float(uni_layer_h)
        self.edge_sharpness = float(edge_sharpness)
        self.device = device


def load_tin_refractive_index(wl_nm, dataset="labo"):
    """Interpolate TiN complex permittivity at wavelength ``wl_nm`` (nm)."""
    import numpy as np
    import pandas as pd

    csv = _INDEX_CSV if dataset == "labo" else _INDEX_CSV.replace("labo", "Beliaev-sputtering")
    df = pd.read_csv(csv).astype(float)
    n = np.interp(wl_nm, df["wl (nm)"].values, df["n"].values)
    k = np.interp(wl_nm, df["wl (nm)"].values, df["k"].values)
    return n, k


def setup(args, sim_dtype=torch.complex64, geo_dtype=None):
    """Build and solve the RCWA simulation for the sine TiN structure.

    Returns ``(sim, perm_map)`` following the original API.
    """
    device = args.device
    if geo_dtype is None:
        geo_dtype = args.wl.dtype

    lx = args.sin_period.detach().item()
    ly = lx

    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = lx
    torcwa.rcwa_geo.Ly = ly
    torcwa.rcwa_geo.nx = args.discretization
    torcwa.rcwa_geo.ny = args.discretization
    torcwa.rcwa_geo.grid()

    eps_air = torch.tensor(1.0, dtype=sim_dtype, device=device)
    n_sub = 1.5
    eps_sub = torch.tensor(n_sub**2, dtype=sim_dtype, device=device)

    wl_nm = args.wl.detach().item()
    n, k = load_tin_refractive_index(wl_nm)
    eps_metal = torch.tensor((n**2 - k**2) + 2j * n * k, dtype=sim_dtype, device=device)

    amplitude = args.sin_amplitude
    period = args.sin_period
    num_layers = 40
    dz = (2 * amplitude) / num_layers

    X, _ = torch.meshgrid(torcwa.rcwa_geo.x, torcwa.rcwa_geo.y, indexing="xy")
    h = amplitude * torch.sin(2 * torch.pi * X / period) + amplitude

    freq = 1.0 / args.wl
    sim = torcwa.rcwa(
        freq=freq,
        order=[0, args.nh],
        lattice=[lx, ly],
        dtype=sim_dtype,
        device=device,
    )

    uniform_layer_height = args.uni_layer_h

    sim.add_input_layer(eps=eps_air)
    sim.add_output_layer(eps=eps_sub)

    sim.set_incident_angle(inc_ang=args.ang * torch.pi / 180, azi_ang=0.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")

    sim.add_layer(thickness=uniform_layer_height, eps=eps_metal)

    base = uniform_layer_height
    sharp = args.edge_sharpness
    for _ in range(num_layers):
        z_mid = base + (_ + 0.5) * dz
        mask = torch.sigmoid(sharp * (h - (z_mid - base)))
        layer_eps = mask * eps_metal + (1 - mask) * eps_air
        sim.add_layer(thickness=dz, eps=layer_eps)

    sim.solve_global_smatrix()
    return sim


def get_s_parameters(sim, orders=((0, 0),), evanscent=1e-3):
    """Return S-parameters for all direction/port/polarization combos.

    ``orders`` is a list of ``[m, n]`` order pairs; each returned value is
    the S-parameter of the selected (ref) order.  A single order pair keeps
    the classic specular response; a list sums the power over all of them
    (see ``get_power_efficiencies``).
    """
    params = {}
    for d in DIRECTIONS:
        for port in PORTS:
            for pol in POLARIZATIONS:
                s = sim.s_parameters(
                    orders=list(orders),
                    direction=d,
                    port=port,
                    polarization=pol,
                    ref_order=[0, 0],
                    power_norm=True,
                    evanscent=evanscent,
                )
                params[f"{d}.{port}.{pol}"] = s
    return params


def _sim_orders(sim):
    """Enumerate the full Fourier basis of a solved simulation."""
    return [
        [int(ox), int(oy)]
        for ox in sim.order_x.tolist()
        for oy in sim.order_y.tolist()
    ]


def _propagating_orders(sim, layer_eps=1.5):
    """Orders that carry power in the output layer (real kz)."""
    orders = _sim_orders(sim)
    m = sim._matching_indices(torch.tensor(orders))
    kz = torch.sqrt(
        layer_eps - sim.Kx_norm_dn[m] ** 2 - sim.Ky_norm_dn[m] ** 2
    )
    prop = (torch.real(kz) > 1e-6) & (torch.abs(torch.imag(kz)) < 1e-6)
    return [o for o, p in zip(orders, prop.tolist()) if p]


def get_power_efficiencies(sim, orders=None, evanscent=1e-3):
    """Total (order-summed) power efficiencies per polarization.

    Returns a dict with keys ``R_xx, R_yy, T_xx, T_yy`` giving the total
    reflected / transmitted power fraction (sum over all *propagating*
    diffraction orders), power-normalized so that R + T <= 1 (strictly
    less for absorbing materials such as TiN).

    Only propagating orders are included: evanescent orders produce NaN
    gradients through TORCWA's power normalization, and carry no power.
    """
    if orders is None:
        orders = _propagating_orders(sim)

    def _power(direction, port, pol):
        s = sim.s_parameters(
            orders=orders,
            direction=direction,
            port=port,
            polarization=pol,
            ref_order=[0, 0],
            power_norm=True,
            evanscent=evanscent,
        )
        return torch.sum(torch.abs(s) ** 2)

    return {
        "R_xx": _power("forward", "reflection", "xx"),
        "T_xx": _power("forward", "transmission", "xx"),
        "R_yy": _power("forward", "reflection", "yy"),
        "T_yy": _power("forward", "transmission", "yy"),
    }