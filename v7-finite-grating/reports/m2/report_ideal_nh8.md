# M2 — learned spectral correction (fast RCWA → MEEP)

- dataset: 384 samples, 24 geometries (18 train / 6 val, split by geometry)
- features (16): period, amplitude, base, mean_scale, std_scale, min_scale, n_missing, missing_index, wl, inv_wl, wl_over_period, amp_over_period, base_over_period, rcwa_R, rcwa_T, rcwa_A
- model: MLP hidden=64, depth=3, residual on (R, T)
- trained 800 epochs, lr=0.001, wd=0.001

## Validation metrics (mean ± std over held-out geometries)

| channel | raw RCWA | corrected | improvement |
|---|---:|---:|---:|
| RMSE R | 0.0974 ± 0.0602 | 0.0041 ± 0.0043 | +95.8% |
| RMSE T | 0.1224 ± 0.0763 | 0.0040 ± 0.0034 | +96.7% |
| RMSE A | 0.0267 ± 0.0146 | 0.0018 ± 0.0005 | +93.2% |

## Pooled validation RMSE

| channel | raw RCWA | corrected |
|---|---:|---:|
| R | 0.1145 | 0.0059 |
| T | 0.1443 | 0.0053 |
| A | 0.0304 | 0.0019 |

## Per-geometry RMSE(A)

| spec | raw | corrected |
|---:|---:|---:|
| 001 | 0.0401 | 0.0024 |
| 005 | 0.0325 | 0.0020 |
| 009 | 0.0082 | 0.0011 |
| 014 | 0.0375 | 0.0023 |
| 015 | 0.0047 | 0.0011 |
| 017 | 0.0374 | 0.0021 |
