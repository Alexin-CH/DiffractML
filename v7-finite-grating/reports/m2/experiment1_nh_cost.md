# Experiment 1 follow-up — ideal unit-cell nh vs cost

- specs: 12 subset of the M2 set; pol=s
- ideal unit cell, grid=10 nm, 40 layers
- reference: MEEP finite grating, de-embedded absorption

| nh | ideal RMSE(A) | time (s / 16-λ spectrum) |
|---:|---:|---:|
| 8 | 0.0233 ± 0.0138 | 4.76 |
| 16 | 0.0232 ± 0.0137 | 6.25 |
| 32 | 0.0232 ± 0.0137 | 13.22 |

For comparison (Experiment 1, same 16-λ spectra): supercell at nh=32 takes ~13.5 s per spectrum, so the unit cell is not cheaper at equal nh.  If a lower nh preserves accuracy here, the ideal backbone does give a genuine speed win.
