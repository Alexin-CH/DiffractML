# v4: MLP wavelength-angle + CNN permittivity map to fields prediction

## Description
Here the goal is to predict the EM fields in and around a 1D sine TiN structure given the wavelength, angle and the permittivity map of the structure.

## Dataset
The dataset is generated using TORCWA, it contains 900 samples:
Range of parameters:
- Wavelength: 400nm - 2000nm
- Angle of incidence: 0 - 70 degrees

Fields are initially sampled on a 2D grid of size 256x500 (x,z).

## Model
The model is make of two parts:
1. A MLP that takes as input the wavelength and angle of incidence and outputs a latent representation.
2. A CNN that takes as input the permittivity map and outputs another latent representation.
3. A CNN decoder that takes as input the sum of the two latent representations and outputs the predicted fields.

![loss](assets/ex_loss.png)
