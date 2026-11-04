# M2 — learned spectral correction (fast RCWA → MEEP)

- dataset: 384 samples, 24 geometries (18 train / 6 val, split by geometry)
- features (16): period, amplitude, base, mean_scale, std_scale, min_scale, n_missing, missing_index, wl, inv_wl, wl_over_period, amp_over_period, base_over_period, rcwa_R, rcwa_T, rcwa_A
- model: MLP hidden=64, depth=3, residual on (R, T)
- trained 800 epochs, lr=0.001, wd=0.001

## Validation metrics (mean ± std over held-out geometries)

| channel | raw RCWA | corrected | improvement |
|---|---:|---:|---:|
| RMSE R | 0.0141 ± 0.0023 | 0.0027 ± 0.0014 | +80.8% |
| RMSE T | 0.0094 ± 0.0026 | 0.0040 ± 0.0016 | +57.8% |
| RMSE A | 0.0131 ± 0.0039 | 0.0019 ± 0.0010 | +85.2% |

## Pooled validation RMSE

| channel | raw RCWA | corrected |
|---|---:|---:|
| R | 0.0143 | 0.0031 |
| T | 0.0098 | 0.0043 |
| A | 0.0137 | 0.0022 |

## Per-geometry RMSE(A)

| spec | raw | corrected |
|---:|---:|---:|
| 001 | 0.0109 | 0.0009 |
| 005 | 0.0211 | 0.0037 |
| 009 | 0.0147 | 0.0011 |
| 014 | 0.0100 | 0.0019 |
| 015 | 0.0119 | 0.0012 |
| 017 | 0.0099 | 0.0027 |
