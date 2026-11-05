# Experiment 2 — differentiable encoder in the RCWA loop

## Objective

Train the hybrid model **end to end** with RCWA inside the loop: an encoder
predicts a solver configuration and a latent, RCWA computes the spectrum, a
residual network corrects it, and the loss backpropagates through TORCWA into
the encoder.

## Setup

- **Backbone:** ideal unit cell, `nh=8`, `nx=200`, 16 layers.
- **Encoder:** shared geometry trunk → latent `z_geo` (16-d) consumed by the
  residual head, plus a config head predicting an effective **amplitude** and a
  small **period** delta.
- **Fixed:** base, padding, `edge_sharpness`, `nh`, `nx`, `n_layers`
  (Experiment 1 showed padding/sharpness are unnecessary).
- **Data:** 24 geometries (18 train / 6 held out by geometry), 16 wavelengths
  each, 8 λ sampled per step, early stopping on held-out RMSE(A).
- **Ablation:** `--no-encoder` freezes `z=0` and uses the input geometry, so
  only the residual is trained.

## Gradient audit of the differentiable solve

| parameter | result |
|---|---|
| effective amplitude | autograd ≈ finite difference (~1.5%) — well conditioned |
| effective period | small, noisy, sign-inconsistent near the operating point |
| `edge_sharpness` | negligible gradient |
| `base` | constant in the dataset |

## Result (held out, mean ± std over 6 geometries)

| variant | RMSE R | RMSE T | RMSE A | train cost |
|---|---:|---:|---:|---:|
| learned encoder (`e2`) | 0.0107 ± 0.0077 | 0.0118 ± 0.0095 | 0.0052 ± 0.0010 | ~39 s/epoch |
| no-encoder ablation (`e2_noenc`) | **0.0067 ± 0.0044** | **0.0082 ± 0.0064** | 0.0053 ± 0.0013 | ~26 s/epoch |
| fixed M2, ideal `nh=8` (E1) | 0.0041 | 0.0040 | **0.0018** | — |

## Interpretation

- **The differentiable loop works:** training converges and gradients reach the
  encoder through TORCWA's eigendecomposition; the gradient test passes.
- **The learned encoder does not help yet.** The no-encoder ablation is better
  on R/T and equal on A, at ~33% lower cost. The encoder adds a noisier
  optimization path without information the residual cannot already read.
- **Why:** only the effective amplitude is well conditioned; the period gradient
  is noisy, and 24 geometries is small for a 16-d latent plus a config head.
- **The residual does the work.** E2's absolute absorption error (0.0052) is
  above the fixed M2 result (0.0018) mainly because it trains on 8 λ/step for
  ≤25 epochs with a different feature set — a budget difference, not a measured
  ceiling.

## Recommendation

- Keep the cheap ideal-cell backbone + residual correction.
- **Freeze the encoder at the input geometry** for now (the current best).
- Revisit the latent encoder with (1) a larger geometry set, (2) an
  amplitude-only config (drop the noisy period), and (3) a full-λ training
  budget so fixed and learned variants are compared at equal headroom.

## Follow-up (Experiment 3)

Experiment 3 ran exactly that follow-up: matched budget, loss and features,
plus an amplitude-only config and a per-ridge encoder redesign. The encoder
still did not beat the fixed-geometry residual at 6, 12 or 18 training
geometries. See [`E3-fair-encoder.md`](E3-fair-encoder.md).

## Reproduce

```bash
# rcwa-v7 env
python scripts/train_e2.py --epochs 25 --n-wl 8 --patience 6 \
    --device cuda --tag e2
python scripts/train_e2.py --epochs 25 --n-wl 8 --patience 6 \
    --device cuda --tag e2_noenc --no-encoder
```

## Artifacts

- `reports/m2/report_e2.md`, `summary_e2.json`
- `reports/m2/report_e2_noenc.md`, `summary_e2_noenc.json`
- code: `src/rcwa_diff.py`, `src/models_e2.py`, `scripts/train_e2.py`,
  `tests/test_e2.py`
