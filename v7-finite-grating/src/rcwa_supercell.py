"""TORCWA padded-supercell baseline for a finite, defective 1D grating.

A finite grating is represented in periodic RCWA by wrapping the aperture in a
supercell of width ``Lx = W + 2 * padding`` and repeating it.  The aperture is
centred in the supercell and surrounded by air-over-substrate padding.

Units: this module works in TORCWA's nanometre units (the geometry is converted
from the µm-based :class:`grating.GratingSpec`).  Material dispersion is taken
from the *same* Drude-Lorentz fit used by MEEP, so both solvers share an
identical TiN model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from grating import GratingSpec

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FIT_DIR = os.path.normpath(os.path.join(_CURRENT_DIR, "..", "data"))


def _fit_cache(dataset: str) -> str:
    return os.path.join(_DEFAULT_FIT_DIR, f"tin_fit_{dataset}.json")


@dataclass
class RCWAConfig:
    """Discretisation / truncation settings for the padded supercell."""

    padding_um: float = 1.0
    nh: int = 16              # Fourier orders in x (order = [nh, 0])
    grid_nm: float = 40.0     # spatial grid spacing over the supercell
    n_layers: int = 40        # vertical discretisation of the TiN profile
    edge_sharpness: float = 100.0
    device: str = "auto"

    def resolve_device(self) -> str:
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def tin_epsilon(wl_um: float, *, tin_params: Optional[dict] = None,
                tin_dataset: str = "beliaev") -> complex:
    """Complex TiN permittivity at ``wl_um`` from the shared Drude-Lorentz fit."""
    if tin_params is None:
        from tin_dispersion import get_or_fit

        tin_params = get_or_fit(tin_dataset, cache_path=_fit_cache(tin_dataset))
    from tin_dispersion import epsilon_model

    f = 1.0 / float(wl_um)
    return complex(epsilon_model(tin_params, np.array([f]))[0])


def _propagating_orders(sim) -> list:
    """Diffraction orders propagating in both the input and output layers."""
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


def _power_efficiencies(sim) -> dict:
    """Total (order-summed) R/T for x- and y-polarised incidence."""
    import torch

    orders = _propagating_orders(sim)

    def _power(direction, port, pol):
        s = sim.s_parameters(
            orders=orders, direction=direction, port=port, polarization=pol,
            ref_order=[0, 0], power_norm=True, evanscent=1e-3,
        )
        return torch.sum(torch.abs(s) ** 2)

    return {
        "R_xx": float(_power("forward", "reflection", "xx")),
        "T_xx": float(_power("forward", "transmission", "xx")),
        "R_yy": float(_power("forward", "reflection", "yy")),
        "T_yy": float(_power("forward", "transmission", "yy")),
    }


def solve_supercell(
    spec: GratingSpec,
    wl_um: float,
    cfg: RCWAConfig,
    *,
    angle_deg: float = 0.0,
    tin_params: Optional[dict] = None,
    tin_dataset: str = "beliaev",
    eps_override: Optional[complex] = None,
) -> dict:
    """Solve one wavelength and return order-summed R/T plus geometry metadata."""
    import torch
    import torcwa

    device = cfg.resolve_device()

    W_um = spec.aperture_width
    Lx_um = W_um + 2.0 * cfg.padding_um
    Lx_nm = Lx_um * 1000.0

    torcwa.rcwa_geo.dtype = torch.float32
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = Lx_nm
    torcwa.rcwa_geo.Ly = Lx_nm
    torcwa.rcwa_geo.nx = max(2 * cfg.nh + 2, int(round(Lx_nm / cfg.grid_nm)))
    torcwa.rcwa_geo.ny = 1  # structure is invariant in y
    torcwa.rcwa_geo.grid()

    # Height profile (nm) over the centred supercell.
    x_nm = torcwa.rcwa_geo.x_grid  # shape (nx, 1)
    x_um_local = (x_nm.cpu().numpy() - Lx_nm / 2.0) / 1000.0
    h_nm = torch.as_tensor(
        spec.height_profile(x_um_local), dtype=torch.float32, device=device
    ) * 1000.0

    eps_air = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device)
    eps_sub = torch.tensor(float(spec.substrate_index) ** 2 + 0.0j,
                           dtype=torch.complex64, device=device)
    eps_val = (eps_override if eps_override is not None
               else tin_epsilon(wl_um, tin_params=tin_params, tin_dataset=tin_dataset))
    eps_tin = torch.tensor(complex(eps_val), dtype=torch.complex64, device=device)

    wl_nm = wl_um * 1000.0
    sim = torcwa.rcwa(
        freq=1.0 / wl_nm, order=[cfg.nh, 0], lattice=[Lx_nm, Lx_nm],
        dtype=torch.complex64, device=device,
    )
    sim.add_input_layer(eps=eps_air)
    sim.add_output_layer(eps=eps_sub)
    sim.set_incident_angle(inc_ang=angle_deg * np.pi / 180.0, azi_ang=0.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")

    H_nm = float(h_nm.max())
    if H_nm > 0.0:
        dz = H_nm / cfg.n_layers
        for k in range(cfg.n_layers):
            z_mid = H_nm - (k + 0.5) * dz  # top -> bottom (input = air side)
            mask = torch.sigmoid(cfg.edge_sharpness * (h_nm - z_mid))
            eps = mask * eps_tin + (1.0 - mask) * eps_air
            sim.add_layer(thickness=dz, eps=eps)

    sim.solve()
    eff = _power_efficiencies(sim)
    eff.update({"Lx_um": Lx_um, "W_um": W_um, "H_nm": H_nm,
                "nx": torcwa.rcwa_geo.nx, "nh": cfg.nh})
    return eff


def spectrum_supercell(
    spec: GratingSpec,
    wl_grid_um,
    cfg: RCWAConfig,
    *,
    angle_deg: float = 0.0,
    pol: str = "s",
    tin_params: Optional[dict] = None,
    tin_dataset: str = "beliaev",
    eps_override: Optional[complex] = None,
) -> dict:
    """RCWA R/T spectrum on ``wl_grid_um`` for one polarisation.

    ``pol="s"`` (E along physical y, out of the x-z plane) maps to TORCWA's
    ``yy``; ``pol="p"`` (E in the x-z plane) maps to ``xx``.
    """
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")
    if tin_params is None:
        from tin_dispersion import get_or_fit

        tin_params = get_or_fit(tin_dataset, cache_path=_fit_cache(tin_dataset))

    R_key = "R_yy" if pol == "s" else "R_xx"
    T_key = "T_yy" if pol == "s" else "T_xx"

    wl_grid_um = np.asarray(wl_grid_um, dtype=float)
    R, T, meta = [], [], None
    for wl in wl_grid_um:
        eff = solve_supercell(
            spec, float(wl), cfg, angle_deg=angle_deg,
            tin_params=tin_params, tin_dataset=tin_dataset,
            eps_override=eps_override,
        )
        R.append(eff[R_key])
        T.append(eff[T_key])
        meta = eff
    return {
        "wl_um": wl_grid_um,
        "R": np.asarray(R),
        "T": np.asarray(T),
        "A": 1.0 - np.asarray(R) - np.asarray(T),
        "Lx_um": meta["Lx_um"],
        "W_um": meta["W_um"],
        "nh": cfg.nh,
        "padding_um": cfg.padding_um,
    }
