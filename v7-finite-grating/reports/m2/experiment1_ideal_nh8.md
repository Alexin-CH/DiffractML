# Experiment 1 — ideal unit cell vs padded supercell

- geometries: 24 (from the M2 dataset)
- RCWA: nh=8, grid=10 nm, 40 layers, pol=s; supercell padding=1 µm
- ideal unit cell = one clean period, no defects (blind to defects)
- supercell = actual defective profile, padded

## Raw RCWA accuracy vs MEEP (RMSE of de-embedded spectra)

| subset | n | ideal RMSE(A) | supercell RMSE(A) | ideal RMSE(R) | supercell RMSE(R) | ideal RMSE(T) | supercell RMSE(T) |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 24 | 0.0236 ± 0.0149 | 0.0148 ± 0.0039 | 0.0880 | 0.0140 | 0.1094 | 0.0104 |
| no_missing | 10 | 0.0062 ± 0.0014 | 0.0135 ± 0.0013 | 0.0125 | 0.0164 | 0.0142 | 0.0108 |
| missing | 14 | 0.0360 ± 0.0028 | 0.0157 ± 0.0047 | 0.1419 | 0.0124 | 0.1774 | 0.0100 |

## Solve cost (seconds per spectrum of 16 wavelengths)

- ideal unit cell: 4.65 s (n=24)
- supercell: 4.53 s (n=3)
- speed-up: 0.97×

Note: RCWA cost is dominated by the Fourier-order eigendecomposition, which depends on `nh`, not on the supercell width. The unit cell is cheaper mainly because it is under-resolved in the supercell for the same `nh`; a fair accuracy-matched comparison may need more orders for the supercell (see the M1 sweep).

## Per-geometry RMSE(A)

| spec | missing | jitter std | ideal | supercell |
|---:|---:|---:|---:|---:|
| 000 | 1 | 0.103 | 0.0358 | 0.0182 |
| 001 | 1 | 0.090 | 0.0401 | 0.0109 |
| 002 | 1 | 0.107 | 0.0366 | 0.0189 |
| 003 | 0 | 0.077 | 0.0066 | 0.0142 |
| 004 | 0 | 0.068 | 0.0089 | 0.0157 |
| 005 | 1 | 0.100 | 0.0325 | 0.0211 |
| 006 | 1 | 0.132 | 0.0344 | 0.0185 |
| 007 | 1 | 0.090 | 0.0323 | 0.0208 |
| 008 | 1 | 0.137 | 0.0319 | 0.0204 |
| 009 | 0 | 0.080 | 0.0082 | 0.0147 |
| 010 | 0 | 0.107 | 0.0072 | 0.0150 |
| 011 | 0 | 0.096 | 0.0051 | 0.0129 |
| 012 | 0 | 0.125 | 0.0048 | 0.0119 |
| 013 | 1 | 0.052 | 0.0396 | 0.0182 |
| 014 | 1 | 0.088 | 0.0375 | 0.0100 |
| 015 | 0 | 0.112 | 0.0047 | 0.0119 |
| 016 | 0 | 0.101 | 0.0055 | 0.0131 |
| 017 | 1 | 0.081 | 0.0374 | 0.0099 |
| 018 | 1 | 0.062 | 0.0381 | 0.0104 |
| 019 | 0 | 0.054 | 0.0057 | 0.0131 |
| 020 | 0 | 0.105 | 0.0050 | 0.0120 |
| 021 | 1 | 0.056 | 0.0392 | 0.0108 |
| 022 | 1 | 0.047 | 0.0325 | 0.0216 |
| 023 | 1 | 0.048 | 0.0363 | 0.0103 |
