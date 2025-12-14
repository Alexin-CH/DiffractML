import os

current_dir = os.path.dirname(os.path.abspath(__file__))

import pandas as pd
import numpy as np

import torch
import torcwa

class RCWAArgs:
    """Class to hold simulation parameters."""
    def __init__(self, wl, ang, nh, discretization, sin_amplitude, sin_period):
        self.wl = wl.requires_grad_()
        self.ang = ang.requires_grad_()
        self.nh = nh
        self.discretization = discretization
        self.sin_amplitude = sin_amplitude.requires_grad_()
        self.sin_period = sin_period.requires_grad_()

def setup(
        args,
        sim_dtype=torch.complex64,
        geo_dtype=torch.float32,
        device='cpu'
    ):
    """
    Setup and run a torcwa RCWA simulation based on the provided arguments.
    Args:
        args: Parsed command-line arguments containing simulation parameters.
            Attributes:
                wl: Wavelength in nm.
                ang: Incidence angle in degrees.
                nh: Number of Fourier harmonics.
                discretization: Grid size in x and y.
                sin_amplitude: Amplitude of sinusoidal corrugation in nm.   
                sin_period: Period of sinusoidal corrugation in nm.
        sim_dtype: Data type for simulation tensors (default: torch.complex64).
        geo_dtype: Data type for geometry tensors (default: torch.float32).
        device: Device to run the simulation on ('cpu' or 'cuda').
    """
    # 3) Unit cell size (nm)
    Lx, Ly = 1000.0, 1000.0
    L = [Lx, Ly]

    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    torcwa.rcwa_geo.Lx = Lx
    torcwa.rcwa_geo.Ly = Ly
    torcwa.rcwa_geo.nx = args.discretization
    torcwa.rcwa_geo.ny = args.discretization
    torcwa.rcwa_geo.grid()

    # 4) Materials
    eps_air   = torch.tensor(1.0, dtype=sim_dtype, device=device)
    n_sub     = 1.44
    eps_sub   = torch.tensor(n_sub**2, dtype=sim_dtype, device=device)
    
    TiN_RefIdx = pd.read_csv(f"{current_dir}/TiN-RefIdx-Beliaev-sputtering.csv").astype(float)
    n = np.interp(
        args.wl.cpu().detach().numpy() / 1000.0,
        TiN_RefIdx['wl (microm)'].values,
        TiN_RefIdx['n'].values
    )
    k = np.interp(
        args.wl.cpu().detach().numpy() / 1000.0,
        TiN_RefIdx['wl (microm)'].values,
        TiN_RefIdx['k'].values
    )

    eps = (n**2 - k**2) + 1j*(2*n*k)
    # print(f"At λ={args.wl.item():.1f} nm, TiN n={n:.3f}, k={k:.3f}, ε={eps:.3f}")
    eps_metal = torch.tensor(eps, dtype=sim_dtype, device=device)

    # 5) Sinusoidal corrugation parameters
    amplitude  = args.sin_amplitude  # nm
    period     = args.sin_period    # nm
    num_layers = 30
    dz         = (2 * amplitude) / num_layers

    # 6) z‐axis and central x‐slice index
    zmax = 1000.0  # nm
    z = torch.linspace(-zmax, zmax, 500, device=device)
    nz = z.numel()
    x_slice = Lx / 2
    x_idx = torch.argmin(torch.abs(torcwa.rcwa_geo.x - x_slice)).item()

    # Precompute height profile h(x,y)
    X, Y = torch.meshgrid(torcwa.rcwa_geo.x,
                          torcwa.rcwa_geo.y,
                          indexing="xy")
    h = amplitude * torch.sin(2 * torch.pi * X / period) + amplitude

    def create_pattern_layer(z_mid, base):
        # Returns a mask = 1 where metal, 0 where air
        return (h >= (z_mid - base)).to(sim_dtype)

    # 7) Build and run RCWA simulation
    freq = 1.0 / args.wl  # in 1/nm
    sim = torcwa.rcwa(freq=freq,
                      order=[0, args.nh],
                      L=L,
                      dtype=sim_dtype,
                      device=device)

    # Prepare permittivity map (z vs y) for diagnostics
    perm_map = torch.zeros((nz, torcwa.rcwa_geo.ny),
                           dtype=sim_dtype,
                           device=device)

    # 7.1) Superstrate (air)
    total_struct = 2 * amplitude + 50.0
    mask_sup = (z >= total_struct)
    perm_map[mask_sup, :] = eps_air
    sim.add_input_layer(eps=eps_air)
    sim.add_output_layer(eps=eps_sub)

    # 7.2) Incident wave and angle
    sim.set_incident_angle(inc_ang=args.ang * torch.pi/180, azi_ang=0.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")

    # 7.3) Uniform metal base layer (50 nm)
    cumul = 0.0
    z_bot, z_top = cumul, cumul + 50.0
    sim.add_layer(thickness=50.0, eps=eps_metal)
    idx = (z >= z_bot) & (z < z_top)
    perm_map[idx, :] = eps_metal
    cumul = z_top

    # 7.4) Sinusoidal layers
    base = cumul
    for _ in range(num_layers):
        z_bot = cumul
        z_top = cumul + dz
        z_mid = 0.5 * (z_bot + z_top)

        mask2d = create_pattern_layer(z_mid, base)
        layer_eps = mask2d * eps_metal + (1 - mask2d) * eps_air

        sim.add_layer(thickness=dz, eps=layer_eps)

        idx = (z >= z_bot) & (z < z_top)
        perm_map[idx, :] = layer_eps[x_idx, :]
        cumul = z_top

    # 7.5) Substrate (SiO₂)
    mask_sub = (z < 0.0)
    perm_map[mask_sub, :] = eps_sub

    # 7.6) Solve S‐matrix and compute reflectance/transmittance
    sim.solve_global_smatrix()

    return sim, perm_map

def get_diffraction_angles(torcwa_simulation):
    inc_angles = []
    azi_angles = []
    orders = torch.arange(10)
    for order in orders:
        inc_angle, azi_angle = torcwa_simulation.diffraction_angle(
            orders=[order, order],
            layer='output',
            unit='radian'
        )
        inc_angles.append(inc_angle)
        azi_angles.append(azi_angle)
    return torch.unique(inc_angles), torch.unique(azi_angles)

def get_S_parameters(torcwa_simulation):
    params = {}
    for d in ['forward', 'backward']:
        for port in ['transmission', 'reflection']:
            for pol in ['xx', 'yx', 'xy', 'yy', 'pp', 'sp', 'ps', 'ss']:
                s_param = torcwa_simulation.S_parameters(
                    orders=[0, 0],
                    direction='forward',
                    port='transmission',
                    polarization='xx',
                    ref_order=[0,0],
                    power_norm=True,
                    evanscent=1e-3
                )
                params[f"{d}.{port}.{pol}"] = s_param.item()
    return params

def export_data_dict(torcwa_simulation):
        """Export relevant data from the torcwa simulation into a dictionary."""
        data_dict = {
            'freq': torcwa_simulation.freq,
            'inc_ang': torcwa_simulation.inc_ang,
            'azi_ang': torcwa_simulation.azi_ang,
            'L': torcwa_simulation.L,
            'Kx_norm_dn': torcwa_simulation.Kx_norm_dn,
            'Ky_norm_dn': torcwa_simulation.Ky_norm_dn,
            'Kx_norm': torcwa_simulation.Kx_norm,
            'Ky_norm': torcwa_simulation.Ky_norm,
            'S': torcwa_simulation.S,
            'C': torcwa_simulation.C,
            'Vi': torcwa_simulation.Vi,
            'Vo': torcwa_simulation.Vo,
            'Vf': torcwa_simulation.Vf,
            'E_i': torcwa_simulation.E_i,
            'eps_in': torcwa_simulation.eps_in,
            'mu_in': torcwa_simulation.mu_in,
            'eps_out': torcwa_simulation.eps_out,
            'mu_out': torcwa_simulation.mu_out,
            'kx_norm': torcwa_simulation.kx_norm,
            'ky_norm': torcwa_simulation.ky_norm,
            'kz_norm': torcwa_simulation.kz_norm,
            'order': torcwa_simulation.order,
        }
        return data_dict
        
            #'thickness': torcwa_simulation.thickness,
            #'layer_N': torcwa_simulation.layer_N,
            #'P': torcwa_simulation.P,
            #'Q': torcwa_simulation.Q,
            #'E_eigvec': torcwa_simulation.E_eigvec,
            #'H_eigvec': torcwa_simulation.H_eigvec,
            #'Cf': torcwa_simulation.Cf,
            #'Cb': torcwa_simulation.Cb,
            #'layer_S11': torcwa_simulation.layer_S11,
            #'layer_S21': torcwa_simulation.layer_S21,
            #'layer_S12': torcwa_simulation.layer_S12,
            #'layer_S22': torcwa_simulation.layer_S22,
            #'Sin': torcwa_simulation.Sin,
            #'Sout': torcwa_simulation.Sout,
            #'order_N': torcwa_simulation.order_N,
            #'kx0_norm': torcwa_simulation.kx0_norm,
            #'ky0_norm': torcwa_simulation.ky0_norm,
            #'angle_layer': torcwa_simulation.angle_layer,
            #'device': torcwa_simulation._device,
            #'dtype': torcwa_simulation._dtype,
            #'eps_conv': torcwa_simulation.eps_conv,
            #'mu_conv': torcwa_simulation.mu_conv,
            #'order_x': torcwa_simulation.order_x,
            #'order_y': torcwa_simulation.order_y,
            #'Gx_norm': torcwa_simulation.Gx_norm,
            #'Gy_norm': torcwa_simulation.Gy_norm,