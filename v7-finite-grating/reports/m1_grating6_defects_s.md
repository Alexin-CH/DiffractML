# M1 benchmark — padded-supercell RCWA vs MEEP finite grating

- ridges: 6, period: 1.2 µm, amplitude: 0.05 µm, base: 0.03 µm
- defects: jitter_std≈0.044, missing=1
- polarisation: s, angle: 0°, substrate n=1.5
- MEEP reference: Lx=11.20 µm (W=7.20 µm), resolution=140.0 px/µm

Primary metric: RMSE of the de-embedded **absorption** spectrum `A=(1-R-T)·Lx/W` (baseline-free). R/T RMSE are after subtracting the bare-substrate Fresnel baseline.

**Best config:** padding=0.5 µm, nh=64, RMSE(A)=0.0124, RMSE(R)=0.0081, RMSE(T)=0.0081.

| padding (µm) | nh | grid (nm) | RMSE A | RMSE R | RMSE T | max|ΔA| | matched res | mean Δλ (µm) | mean Δwidth (µm) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 8 | 10 | 0.0302 | 0.0065 | 0.0362 | 0.0352 | 0 | nan | nan |
| 0.5 | 16 | 10 | 0.0188 | 0.0050 | 0.0165 | 0.0237 | 0 | nan | nan |
| 0.5 | 32 | 10 | 0.0135 | 0.0075 | 0.0092 | 0.0194 | 0 | nan | nan |
| 0.5 | 64 | 10 | 0.0124 | 0.0081 | 0.0081 | 0.0183 | 0 | nan | nan |
| 1 | 8 | 10 | 0.0339 | 0.0106 | 0.0441 | 0.0386 | 0 | nan | nan |
| 1 | 16 | 10 | 0.0210 | 0.0054 | 0.0173 | 0.0272 | 0 | nan | nan |
| 1 | 32 | 10 | 0.0147 | 0.0082 | 0.0093 | 0.0218 | 0 | nan | nan |
| 1 | 64 | 10 | 0.0134 | 0.0091 | 0.0081 | 0.0205 | 0 | nan | nan |
| 2 | 8 | 10 | 0.0679 | 0.0149 | 0.0545 | 0.0750 | 0 | nan | nan |
| 2 | 16 | 10 | 0.0323 | 0.0063 | 0.0382 | 0.0375 | 0 | nan | nan |
| 2 | 32 | 10 | 0.0163 | 0.0077 | 0.0104 | 0.0241 | 0 | nan | nan |
| 2 | 64 | 10 | 0.0138 | 0.0094 | 0.0074 | 0.0218 | 0 | nan | nan |
| 4 | 8 | 10 | 0.0936 | 0.0264 | 0.1176 | 0.1048 | 0 | nan | nan |
| 4 | 16 | 10 | 0.0339 | 0.0115 | 0.0448 | 0.0397 | 0 | nan | nan |
| 4 | 32 | 10 | 0.0202 | 0.0058 | 0.0156 | 0.0287 | 0 | nan | nan |
| 4 | 64 | 10 | 0.0150 | 0.0087 | 0.0088 | 0.0241 | 0 | nan | nan |
