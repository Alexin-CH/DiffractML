# Experiment 3 — fair encoder comparison and redesign

## Fair ladder (same data, loss, optimiser and schedule)

All arms: ideal unit cell `nh=8`, precomputed RCWA, 18/6 geometry split, MSE on standardised `(dR, dT)`, batch 64, AdamW lr=1e-3 wd=1e-3.

| encoder | RMSE R | RMSE T | RMSE A | params | wall |
|---|---:|---:|---:|---:|---:|
| none | 0.0065 ± 0.0020 | 0.0072 ± 0.0021 | 0.0021 ± 0.0003 | 35458 | 5.2s |
| global | 0.0055 ± 0.0009 | 0.0065 ± 0.0008 | 0.0024 ± 0.0001 | 57234 | 3.0s |
| ridge | 0.0053 ± 0.0017 | 0.0062 ± 0.0022 | 0.0021 ± 0.0005 | 90130 | 7.6s |

## Sample efficiency (held-out RMSE(A), mean ± std over seeds)

| n_train | fixed | global | ridge |
|---:|---:|---:|---:|
| 6 | 0.0095 ± 0.0047 | 0.0077 ± 0.0037 | 0.0095 ± 0.0051 |
| 12 | 0.0026 ± 0.0004 | 0.0026 ± 0.0003 | 0.0026 ± 0.0002 |
| 18 | 0.0021 ± 0.0003 | 0.0024 ± 0.0001 | 0.0021 ± 0.0005 |

