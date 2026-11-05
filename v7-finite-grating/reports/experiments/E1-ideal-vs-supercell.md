# Experiment 1 — does the solver need a padded supercell?

## Objective

Test whether the expensive padded-supercell RCWA is necessary, or whether a
single **ideal period** is enough once a learned correction is applied.

## Setup

- Reuse the 24 MEEP references from M2 (finite defective gratings).
- **Ideal unit cell:** one clean period (`n_ridges=1`, no defects), same
  period/amplitude/base; periodic in x.
- **Supercell:** the actual defective profile in a padded supercell
  (padding 1.0 µm, `nh=32`).
- **Reference:** MEEP finite grating, de-embedded to the aperture.
- Two questions: (a) raw RCWA error vs MEEP, (b) cost at equal and at
  accuracy-matched settings.

## Raw accuracy vs MEEP

| subset | n | ideal RMSE(A) | supercell RMSE(A) |
|---|---:|---:|---:|
| all | 24 | 0.0235 | **0.0148** |
| jitter-only (no missing ridge) | 10 | **0.0062** | 0.0135 |
| missing ridge | 14 | 0.0359 | **0.0157** |

- **Jitter-only:** the ideal cell is *better* — the `padding=1.0, nh=32`
  supercell is under-resolved and adds error.
- **Missing ridge:** the supercell wins, because the ideal cell cannot see the
  defect at all.

## Cost

- At **equal `nh`** the two are identical: 13.47 s vs 13.50 s per 16-λ
  spectrum. RCWA cost is dominated by the Fourier-order eigendecomposition,
  not the supercell width.
- **But the ideal cell converges at `nh=8`:**

| nh | ideal RMSE(A) | time (s / 16-λ spectrum) |
|---:|---:|---:|
| 8 | 0.0233 | **4.76** |
| 16 | 0.0232 | 6.25 |
| 32 | 0.0232 | 13.22 |

The supercell needs `nh=32` (13.5 s) to resolve its ridges. The cheap operating
point is therefore the ideal cell at `nh=8`.

## End-to-end corrected result

| backbone | solve cost | corrected RMSE(A) | corrected RMSE(R) | corrected RMSE(T) |
|---|---:|---:|---:|---:|
| supercell, nh=32 | 13.5 s | 0.0019 | **0.0027** | 0.0040 |
| ideal, nh=32 | 13.2 s | 0.0018 | 0.0041 | 0.0040 |
| **ideal, nh=8** | **4.65 s** | **0.0018** | 0.0041 | 0.0040 |

## Interpretation

- The supercell is **not required** for the corrected absorption accuracy: the
  ideal cell at `nh=8` plus the residual matches the supercell (0.0018 vs
  0.0019) at **~2.9× lower solve cost**.
- The supercell retains a small edge on R (0.0027 vs 0.0041). This is the gap
  the encoder work in Experiment 2 was meant to close.
- The "one perfect period + NNs" route is viable and cheaper; the supercell is
  useful mainly when raw (uncorrected) R accuracy matters or when defects are
  severe.

## Reproduce

```bash
# meep-v7 env — references already exist in reports/m2/data
# rcwa-v7 env
python scripts/benchmark_ideal_vs_supercell.py --nh 32 \
    --build-ideal-dataset --device cuda
python scripts/train_m2.py --data reports/m2/dataset_ideal.npz --tag ideal \
    --epochs 800 --hidden 64 --depth 3 --weight-decay 1e-3 --patience 150
python scripts/sweep_ideal_nh.py --device cuda --n-specs 12 --nhs 8 16 32
python scripts/benchmark_ideal_vs_supercell.py --nh 8 \
    --build-ideal-dataset --ideal-data-out reports/m2/dataset_ideal_nh8.npz
python scripts/train_m2.py --data reports/m2/dataset_ideal_nh8.npz --tag ideal_nh8 \
    --epochs 800 --hidden 64 --depth 3 --weight-decay 1e-3 --patience 150
```

## Artifacts

- `reports/m2/experiment1_ideal_vs_supercell.md`
- `reports/m2/experiment1_nh_cost.md`
- `reports/m2/report_ideal.md`, `summary_ideal.json`
- `reports/m2/dataset_ideal.npz`, `dataset_ideal_nh8.npz`
- code: `scripts/benchmark_ideal_vs_supercell.py`, `scripts/sweep_ideal_nh.py`
