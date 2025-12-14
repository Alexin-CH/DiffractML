# SinTiN Torcwa

Simulation of the optical response of diffractive structures using TORCWA.

Here the structure is a sine grating of TiN on a SiO2 substrate.

## Table of numical parameters :

| Parameter               | Value     | Unit          | Description                                     |
|-------------------------|-----------|---------------|-------------------------------------------------|
| Wavelength range        | 500-2000  | nm            | Wavelength sweep range                          |
| Incident angle          | 30        | °             | Angle of incidence                              |
| Grating period          | 1000      | nm            | Period of the sine grating                      |
| Grating height          | 110       | nm            | Height of the sine grating (amplitude is 55 nm) |

## Eval orders :

Here only need to set y order as this is a 1D grating

![eval_orders](assets/time_vs_orders.png)


## First order response

| xx polarization at 30° incidence            | yy polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![xx_30deg](src/img/spectrum_xx_30deg.png)  | ![yy_30deg](src/img/spectrum_yy_30deg.png)  |

| xy polarization at 30° incidence            | yx polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![xy_30deg](src/img/spectrum_xy_30deg.png)  | ![yx_30deg](src/img/spectrum_yx_30deg.png)  |

* * *

| pp polarization at 30° incidence            | ss polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![pp_30deg](src/img/spectrum_pp_30deg.png)  | ![ss_30deg](src/img/spectrum_ss_30deg.png)  |

| ps polarization at 30° incidence            | sp polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![ps_30deg](src/img/spectrum_ps_30deg.png)  | ![sp_30deg](src/img/spectrum_sp_30deg.png)  |
