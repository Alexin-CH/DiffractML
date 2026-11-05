# Experiment 2 — end-to-end encoder → RCWA → residual

- backbone: ideal unit cell, nh=8, nx=200, 16 layers
- encoder latent=16, hidden=64; learned cfg (amp, period)
- 18 train / 6 val geometries, early stop (patience 6), 8 λ/step

## Held-out metrics (mean ± std over geometries)

| channel | raw RCWA | corrected |
|---|---:|---:|
| RMSE R | 0.0970 ± 0.0594 | 0.0107 ± 0.0077 |
| RMSE T | 0.1220 ± 0.0761 | 0.0118 ± 0.0095 |
| RMSE A | 0.0270 ± 0.0149 | 0.0052 ± 0.0010 |

## Per-geometry RMSE(A)

| spec | raw | corrected |
|---:|---:|---:|
| 001 | 0.0406 | 0.0062 |
| 005 | 0.0328 | 0.0052 |
| 009 | 0.0078 | 0.0066 |
| 014 | 0.0375 | 0.0052 |
| 015 | 0.0047 | 0.0036 |
| 017 | 0.0386 | 0.0044 |
