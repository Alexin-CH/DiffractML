# SinTiN TORCWA Simulations

Simulation of the optical response of diffractive structures using TORCWA.

Here the structure is a sine grating of TiN on a SiO2 substrate.

## Eval orders :

Here only need to set y order as this is a 1D grating

![eval_orders](assets/time_vs_orders.png)

* * *
* * *

## Simulation n°1 : Sine grating of TiN on SiO2 substrate, period 1000 nm, height 110 nm, incident angle 30°

### Table of parameters :

| Parameter               | Value     | Unit          | Description                                     |
|-------------------------|-----------|---------------|-------------------------------------------------|
| Wavelength range        | 500-2000  | nm            | Wavelength sweep range                          |
| Incident angle          | 30        | °             | Angle of incidence                              |
| Grating period          | 1000      | nm            | Period of the sine grating                      |
| Grating height          | 110       | nm            | Height of the sine grating (amplitude is 55 nm) |


### 0th order response

| xx polarization at 30° incidence            | yy polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![xx_30deg](src/img/spectrum_xx_30deg_1000_55.png)  | ![yy_30deg](src/img/spectrum_yy_30deg_1000_55.png)  |

| xy polarization at 30° incidence            | yx polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![xy_30deg](src/img/spectrum_xy_30deg_1000_55.png)  | ![yx_30deg](src/img/spectrum_yx_30deg_1000_55.png)  |

* * *

| pp polarization at 30° incidence            | ss polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![pp_30deg](src/img/spectrum_pp_30deg_1000_55.png)  | ![ss_30deg](src/img/spectrum_ss_30deg_1000_55.png)  |

| ps polarization at 30° incidence            | sp polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![ps_30deg](src/img/spectrum_ps_30deg_1000_55.png)  | ![sp_30deg](src/img/spectrum_sp_30deg_1000_55.png)  |

* * *
* * *

## Simulation n°2 : Sine grating of TiN on substrate (n=1.5), period 450 nm, height 90 nm, incident angle 30°

### Table of parameters :

| Parameter               | Value     | Unit          | Description                                     |
|-------------------------|-----------|---------------|-------------------------------------------------|
| Wavelength range        | 500-2000  | nm            | Wavelength sweep range                          |
| Incident angle          | 30        | °             | Angle of incidence                              |
| Grating period          | 450       | nm            | Period of the sine grating                      |
| Grating height          | 90        | nm            | Height of the sine grating (amplitude is 45 nm) |

### 0th order response

| xx polarization at 30° incidence            | yy polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![xx_30deg_450](src/img/spectrum_xx_30deg_450_45.png)  | ![yy_30deg_450](src/img/spectrum_yy_30deg_450_45.png)  |
| xy polarization at 30° incidence            | yx polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![xy_30deg_450](src/img/spectrum_xy_30deg_450_45.png)  | ![yx_30deg_450](src/img/spectrum_yx_30deg_450_45.png)  |

* * *

| pp polarization at 30° incidence            | ss polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![pp_30deg_450](src/img/spectrum_pp_30deg_450_45.png)  | ![ss_30deg_450](src/img/spectrum_ss_30deg_450_45.png)  |
| ps polarization at 30° incidence            | sp polarization at 30° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![ps_30deg_450](src/img/spectrum_ps_30deg_450_45.png)  | ![sp_30deg_450](src/img/spectrum_sp_30deg_450_45.png)  |

* * *
* * *

## Simulation n°3 : Sine grating of TiN on substrate (n=1.5), period 450 nm, height 90 nm, incident angle 25°

### Table of parameters :
| Parameter               | Value     | Unit          | Description                                     |
|-------------------------|-----------|---------------|-------------------------------------------------|
| Wavelength range        | 500-2000  | nm            | Wavelength sweep range                          |
| Incident angle          | 25        | °             | Angle of incidence                              |
| Grating period          | 450       | nm            | Period of the sine grating                      |
| Grating height          | 90        | nm            | Height of the sine grating (amplitude is 45 nm) |

### 0th order response

| xx polarization at 25° incidence            | yy polarization at 25° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![xx_25deg_450](src/img/spectrum_xx_25deg_450_45.png)  | ![yy_25deg_450](src/img/spectrum_yy_25deg_450_45.png)  |
| xy polarization at 25° incidence            | yx polarization at 25° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![xy_25deg_450](src/img/spectrum_xy_25deg_450_45.png)  | ![yx_25deg_450](src/img/spectrum_yx_25deg_450_45.png)  |

* * *

| pp polarization at 25° incidence            | ss polarization at 25° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![pp_25deg_450](src/img/spectrum_pp_25deg_450_45.png)  | ![ss_25deg_450](src/img/spectrum_ss_25deg_450_45.png)  |
| ps polarization at 25° incidence            | sp polarization at 25° incidence            |
|---------------------------------------------|---------------------------------------------|
| ![ps_25deg_450](src/img/spectrum_ps_25deg_450_45.png)  | ![sp_25deg_450](src/img/spectrum_sp_25deg_450_45.png)  |

