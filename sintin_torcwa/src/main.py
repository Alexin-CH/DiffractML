import os

current_dir = os.path.dirname(os.path.abspath(__file__))

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

from rcwa import setup, RCWAArgs

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simulate_spectrum(wl_list, angle_deg=torch.tensor(30, device=device), order=3):
    """Compute reflection and transmission spectra over wavelengths at a given angle."""
    results = {'xx':[], 'yx':[], 'xy':[], 'yy':[], 'pp':[], 'sp':[], 'ps':[], 'ss':[]}
    tqdm_wl = tqdm(wl_list)
    
    for wl in tqdm_wl:
        tqdm_wl.set_description(f"Simulating λ={wl:.1f} nm")
        args = RCWAArgs(
            wl=wl,
            ang=angle_deg,
            nh=order,
            discretization=2**12,
            sin_amplitude=torch.tensor(55.0, device=device),
            sin_period=torch.tensor(1000.0, device=device)
        )

        # Setup and run simulation
        sim, _  = setup(args=args, device = device)

        # Get Reflextion and Transmission coefficients (order 0)
        for pol in ['xx', 'yx', 'xy', 'yy', 'pp', 'sp', 'ps', 'ss']:
            RF0 = sim.S_parameters(
                orders=[0,0],
                direction='f',
                port='reflection',
                polarization=pol,
                ref_order=[0,0]
            )
            RB0 = sim.S_parameters(
                orders=[0,0],
                direction='b',  
                port='reflection',
                polarization=pol,
                ref_order=[0,0]
            )
            TF0 = sim.S_parameters(
                orders=[0,0],
                direction='f',
                port='transmission',
                polarization=pol,
                ref_order=[0,0]
            )
            TB0 = sim.S_parameters(
                orders=[0,0],
                direction='b',
                port='transmission',
                polarization=pol,
                ref_order=[0,0]
            )

            results[pol].append({
                'wl': wl.detach().cpu().item(),
                'RF0': RF0.abs().detach().cpu().item() ** 2,
                'RB0': RB0.abs().detach().cpu().item() ** 2,
                'TF0': TF0.abs().detach().cpu().item() ** 2,
                'TB0': TB0.abs().detach().cpu().item() ** 2,
            })
    return results, sim

# Example: compute spectrum at 30° incidence
wavelengths = torch.linspace(500., 2000., 100)
inc_angle_deg = torch.tensor(30., device=device)
results, sim = simulate_spectrum(wavelengths, angle_deg=inc_angle_deg, order=4)

wavelengths = wavelengths.cpu().detach()  # Move to CPU for plotting

# Plot reflection and transmission vs wavelength
if True:
    for idx, pol in enumerate(['xx', 'yy', 'pp', 'sp', 'ps', 'ss']):
        RF0 = torch.tensor([entry['RF0'] for entry in results[pol]])
        TF0 = torch.tensor([entry['TF0'] for entry in results[pol]])
        RB0 = torch.tensor([entry['RB0'] for entry in results[pol]])
        TB0 = torch.tensor([entry['TB0'] for entry in results[pol]])

        plt.figure(figsize=(10,6))
        plt.plot(wavelengths, RF0, label='Reflection Forward Order 0')
        plt.plot(wavelengths, TF0, label='Transmission Forward Order 0')
        plt.plot(wavelengths, RB0, label='Reflection Backward Order 0')
        plt.plot(wavelengths, TB0, label='Transmission Backward Order 0')
        # plt.plot(wavelengths, A_spec[:, idx], label='Absorption (A)')  
        plt.title(f"Reflectance and Transmittance (θ={inc_angle_deg}°) at {pol} polarization")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Efficiency")
        plt.ylim(0, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{current_dir}/img/spectrum_{pol}_{inc_angle_deg:.0f}deg.png", dpi=300)
        plt.pause(1)

if False:
    diffraction_angles_in = []
    diffraction_angles_out = []
    for order in torch.arange(0, 10):
        angles_in = sim.diffraction_angle(orders=[order, order], layer='in')[0].cpu().detach()
        angles_out = sim.diffraction_angle(orders=[order, order], layer='out')[0].cpu().detach()
        diffraction_angles_in.append(angles_in)
        diffraction_angles_out.append(angles_out)

    plt.figure(figsize=(6,4))
    plt.plot(np.arange(0,10), diffraction_angles_in, 'o-', label='Incident side')
    plt.plot(np.arange(0,10), diffraction_angles_out, 's--', label='Transmission side')
    plt.title(f"Diffraction Angles vs Order (θ={inc_angle_deg}°)")
    plt.xlabel("Diffracted Order")
    plt.ylabel("Diffraction Angle (degrees)")
    plt.legend()
    plt.grid()
    plt.show()

if False:
    # View XZ-plane fields and export
    sim.source_planewave(amplitude=[1.,0],direction='f')

    x_axis = torcwa.rcwa_geo.x.cpu()
    z_axis = z.cpu()

    t0 = time.time()
    [Ex, Ey, Ez], [Hx, Hy, Hz] = sim.field_yz(torcwa.rcwa_geo.y,z,L[1]/2)
    print(f"Field calculation took {time.time()-t0:.2f} seconds")
    Enorm = torch.sqrt(torch.abs(Ex)**2 + torch.abs(Ey)**2 + torch.abs(Ez)**2)
    Hnorm = torch.sqrt(torch.abs(Hx)**2 + torch.abs(Hy)**2 + torch.abs(Hz)**2)

    print(f"Fields at {1/sim.freq.abs().cpu():.2f}nm")

    fig, axes = plt.subplots(figsize=(10,12),nrows=2,ncols=4)
    im0 = axes[0,0].imshow(torch.transpose(Enorm,-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[0,0].set(title='E norm',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im1 = axes[0,1].imshow(torch.transpose(torch.abs(Ex),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[0,1].set(title='Ex abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im2 = axes[0,2].imshow(torch.transpose(torch.abs(Ey),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[0,2].set(title='Ey abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im3 = axes[0,3].imshow(torch.transpose(torch.abs(Ez),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[0,3].set(title='Ez abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    
    im4 = axes[1,0].imshow(torch.transpose(Hnorm,-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[1,0].set(title='H norm',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im5 = axes[1,1].imshow(torch.transpose(torch.abs(Hx),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[1,1].set(title='Hx abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im6 = axes[1,2].imshow(torch.transpose(torch.abs(Hy),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[1,2].set(title='Hy abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    im7 = axes[1,3].imshow(torch.transpose(torch.abs(Hz),-2,-1).cpu(),cmap='jet',origin='lower',extent=[x_axis[0],x_axis[-1],z_axis[0],z_axis[-1]])
    axes[1,3].set(title='Hz abs',xlim=(0,L[0]),xlabel='y (nm)',ylim=(z_axis[0],z_axis[-1]),ylabel='z (nm)')
    
    fig.colorbar(im0,ax=axes[0,0])
    fig.colorbar(im1,ax=axes[0,1])
    fig.colorbar(im2,ax=axes[0,2])
    fig.colorbar(im3,ax=axes[0,3])
    fig.colorbar(im4,ax=axes[1,0])
    fig.colorbar(im5,ax=axes[1,1])
    fig.colorbar(im6,ax=axes[1,2])
    fig.colorbar(im7,ax=axes[1,3])
    plt.tight_layout()
    plt.show()

