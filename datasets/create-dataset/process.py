import os
import argparse
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import sintin_torcwa.simulation as rcwa

def parse():
    parser = argparse.ArgumentParser(
        description="Generate and save one RCWA sample for a single wavelength and angle"
    )
    parser.add_argument("--wl", type=float, default=1500.0,
                        help="Wavelength in nm")
    parser.add_argument("--ang", type=float, default=30.0,
                        help="Incidence angle in degrees")
    parser.add_argument("--nh", type=int, default=50,
                        help="Number of Fourier harmonics")
    parser.add_argument("--discretization", type=int, default=128,
                        help="Grid size in x and y")
    parser.add_argument("--sin_amplitude", type=float, default=55.0,
                        help="Amplitude of sinusoidal corrugation in nm")
    parser.add_argument("--sin_period", type=float, default=1000.0,
                        help="Period of sinusoidal corrugation in nm")
    parser.add_argument("--perm_map", type=bool, default=False,
                        help="Create permitivitty map condition")
    parser.add_argument("--filename", type=str, default="sample.json",
                        help="Output filename (default: sample.pt)")
    
    args = parser.parse_args()
    return args

def main(args):
    t0 = time.time()

    filepath = args.filename
    amplitude = args.sin_amplitude
    period = args.sin_period

    sim, perm_map = rcwa.setup(args, device='cpu')
    #sample = export_data_dict(sim)
    inc_angles, azi_angles = rcwa.get_diffraction_angles(sim)
    s_params = rcwa.get_S_parameters(sim)

    sample = {}
    sample["wl"] = args.wl
    sample["ang"] = args.ang
    sample["amp"] = args.sin_amplitude
    sample["per"] = args.sin_period
    for k in s_params:
        sample[k] = s_params[k]
    sample['diffraction.inc.angles'] = inc_angles
    sample['diffraction.azi.angles'] = azi_angles

    path = filepath.split('/')[:-1]

    out_dir = os.path.join(*path) if path else '.'
    os.makedirs(out_dir, exist_ok=True)

    filename = filepath.split('/')[-1]

    # Save permittivity map if not already saved
    perm_map_title = f"{filename.split('.')[0]}.perm_map.amp{int(amplitude)}_per{int(period)}.pt"
    if args.perm_map and not os.path.exists(os.path.join(out_dir, perm_map_title)):
        torch.save(perm_map.cpu(), os.path.join(out_dir, perm_map_title))
        print(f"({time.time()-t0:.2f}s) Permittivity map saved to: {os.path.join(out_dir, perm_map_title)}")
    
    # Save simulation data
    sample = pd.DataFrame([sample])
    output_path = os.path.join(out_dir, filename)

    sample.to_json(output_path)
    print(f"({time.time()-t0:.2f}s) Sample saved to: {output_path}")

if __name__ == "__main__":
    args = parse()
    main(args)
