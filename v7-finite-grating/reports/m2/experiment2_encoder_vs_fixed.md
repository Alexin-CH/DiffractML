# Experiment 2 — differentiable encoder vs fixed geometry

## Setup

- **Backbone:** ideal unit cell, `nh=8`, `nx=200`, 16 layers, one warm start per
  geometry; RCWA runs inside the training loop.
- **Encoder:** shared geometry trunk → latent `z_geo` (16-d) consumed by the
  residual head, plus a config head predicting an effective **amplitude** and a
  small effective **period** delta. `base`, `padding`, `sharpness`, `nh`, `nx`,
  `n_layers` are fixed (Experiment 1 showed padding/sharpness are unnecessary).
- **Gradient audit of `solve_unit_cell`:**
  - amplitude: autograd ≈ finite-difference (~1.5% agreement) — well conditioned;
  - period: small, noisy, sign-inconsistent near the operating point;
  - `edge_sharpness` and `base`: negligible / constant.
- **Data:** 24 geometries (18 train / 6 held out by geometry), 16 wavelengths
  each, 8 λ sampled per step, early stopping on held-out RMSE(A).

## Held-out result (mean ± std over the 6 held-out geometries)

| variant | RMSE R | RMSE T | RMSE A | train cost |
|---|---:|---:|---:|---:|
| learned encoder (`e2`) | 0.0107 ± 0.0077 | 0.0118 ± 0.0095 | 0.0052 ± 0.0010 | ~39 s/epoch |
| no-encoder ablation (`e2_noenc`) | **0.0067 ± 0.0044** | **0.0082 ± 0.0064** | 0.0053 ± 0.0013 | ~26 s/epoch |
| fixed M2, ideal `nh=8` (Experiment 1) | 0.0041 | 0.0040 | **0.0018** | — |

## Interpretation

- **The differentiable loop works.** Training converges and gradients reach the
  encoder through TORCWA's eigendecomposition; the gradient test passes.
- **The learned encoder does not help yet.** The no-encoder ablation is better
  on R and T and equal on A, at ~33% lower cost. The encoder introduces a
  noisier optimization path without adding information the residual cannot
  already read from the descriptors + RCWA spectrum.
- **Why:** only the effective amplitude is well conditioned; the period
  gradient is noisy at this operating point, and 24 geometries is small for a
  16-d latent + config head.
- **The residual does the work.** The E2 absolute absorption error (0.0052) is
  above the fixed M2 result (0.0018) mainly because E2 trains on 8 λ/step for
  ≤25 epochs with a different feature set — a training-budget difference, not a
  measured ceiling.

## Recommendation

Keep the cheap ideal-unit-cell backbone and the residual correction; **freeze
the encoder at the input geometry** for now. Revisit the encoder when:

1. the geometry set is large enough to identify a 16-d latent;
2. the period prediction is dropped or preconditioned (amplitude-only config);
3. a longer/full-λ training budget is used so the fixed and learned variants
   are compared at equal accuracy headroom.

Artifacts: `report_e2.md`, `report_e2_noenc.md`, `summary_e2.json`,
`summary_e2_noenc.json`.
