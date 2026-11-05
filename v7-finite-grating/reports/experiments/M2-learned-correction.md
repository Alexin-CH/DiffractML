# M2 — learned spectral correction

## Objective

Learn the residual between the fast RCWA spectrum and the MEEP reference, so a
cheap solve plus a small network approximates the finite-grating response.

## Setup

- **Geometries:** 24 seeded finite gratings (18 train / 6 held out, split by
  geometry), 16 wavelengths each over 0.7–1.15 µm.
- **Fast RCWA:** padded supercell, padding 1.0 µm, `nh=32`, grid 10 nm,
  40 layers. (Experiment 1 later showed an ideal unit cell at `nh=8` is
  cheaper; M2 predates that.)
- **De-embedding:** both solvers are normalised to the aperture:
  `A = (1−R−T)·Lx/W`, `R = R₀ + (R−R₀)·Lx/W`.
- **Features (16):** period, amplitude, base, mean/std/min ridge scale,
  number and position of a missing ridge, λ, 1/λ, λ/period, amplitude/period,
  base/period, and the fast RCWA (R, T, A).
- **Target:** residual `[MEEP_R − RCWA_R, MEEP_T − RCWA_T]`; `A = 1−R−T` is
  derived, so the corrected spectrum sums to one.
- **Model:** MLP hidden 64, depth 3, residual MSE on standardised targets,
  early stopping.

## Result (held out, mean ± std over 6 geometries)

| channel | raw RCWA | corrected | improvement |
|---|---:|---:|---:|
| RMSE R | 0.0141 ± 0.0023 | 0.0027 ± 0.0014 | +80.8% |
| RMSE T | 0.0094 ± 0.0026 | 0.0040 ± 0.0016 | +57.8% |
| RMSE A | 0.0131 ± 0.0039 | 0.0019 ± 0.0010 | +85.2% |

Pooled validation: R 0.0143 → 0.0031, T 0.0098 → 0.0043, A 0.0137 → 0.0022.

Per-geometry RMSE(A): every held-out geometry improves, e.g. 0.0109→0.0009,
0.0211→0.0037, 0.0147→0.0011, 0.0100→0.0019, 0.0119→0.0012, 0.0099→0.0027.

## Interpretation

- The correction generalises across **unseen geometries** and cuts the
  absorption error by ~85%.
- This is the proof of concept for the hybrid idea: a physics backbone plus a
  learned residual.

## Reproduce

```bash
# meep-v7 env — references
python scripts/generate_m2_references.py --n-specs 24 --resolution 105 \
    --wl-range 0.7 1.15 --n-freq 16 --pol s

# rcwa-v7 env — dataset, training
python scripts/build_m2_dataset.py --n-specs 24 --pol s --device cuda
python scripts/train_m2.py --epochs 800 --hidden 64 --depth 3 \
    --weight-decay 1e-3 --patience 150 --tag m2
```

## Artifacts

- `reports/m2/report_m2.md`, `summary_m2.json`, `dataset.npz`
- `reports/m2/data/` (24 MEEP references)
- code: `src/m2_data.py`, `src/models_m2.py`, `scripts/train_m2.py`
