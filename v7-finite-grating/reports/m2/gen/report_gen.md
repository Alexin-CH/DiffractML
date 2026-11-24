# M2 — learned spectral correction (fast RCWA → MEEP)

- dataset: 512 samples, 8 geometries (6 train / 2 val, split by geometry)
- features (19): period, amplitude, base, mean_scale, std_scale, min_scale, n_missing, missing_index, wl, inv_wl, wl_over_period, amp_over_period, base_over_period, rcwa_R, rcwa_T, rcwa_A, sin_theta, cos_theta, pol_p
- model: MLP hidden=128, depth=4, residual on (R, T)
- trained 800 epochs, lr=0.001, wd=0.001

## Validation metrics (mean ± std over held-out geometries)

| channel | raw RCWA | corrected | improvement |
|---|---:|---:|---:|
| RMSE R | 0.0360 ± 0.0012 | 0.0105 ± 0.0041 | +70.8% |
| RMSE T | 0.0576 ± 0.0099 | 0.0097 ± 0.0047 | +83.2% |
| RMSE A | 0.0635 ± 0.0159 | 0.0055 ± 0.0026 | +91.3% |

## Pooled validation RMSE

| channel | raw RCWA | corrected |
|---|---:|---:|
| R | 0.0360 | 0.0113 |
| T | 0.0584 | 0.0108 |
| A | 0.0655 | 0.0061 |

## Per-geometry RMSE(A)

| spec | raw | corrected |
|---:|---:|---:|
| 003 | 0.0476 | 0.0029 |
| 022 | 0.0794 | 0.0082 |
