# Experiment 2 — end-to-end encoder → RCWA → residual

- backbone: ideal unit cell, nh=8, nx=200, 16 layers
- encoder latent=16, hidden=64; NO-ENCODER ablation
- 18 train / 6 val geometries, early stop (patience 6), 8 λ/step

## Held-out metrics (mean ± std over geometries)

| channel | raw RCWA | corrected |
|---|---:|---:|
| RMSE R | 0.0972 ± 0.0596 | 0.0067 ± 0.0044 |
| RMSE T | 0.1224 ± 0.0764 | 0.0082 ± 0.0064 |
| RMSE A | 0.0272 ± 0.0148 | 0.0053 ± 0.0013 |

## Per-geometry RMSE(A)

| spec | raw | corrected |
|---:|---:|---:|
| 001 | 0.0406 | 0.0068 |
| 005 | 0.0336 | 0.0061 |
| 009 | 0.0083 | 0.0068 |
| 014 | 0.0372 | 0.0048 |
| 015 | 0.0048 | 0.0036 |
| 017 | 0.0386 | 0.0038 |
